"""Helper functions for ``check_student_solution.py``.

Students do not need to edit this file.  It keeps the main checker script
shorter by collecting small utilities for:

- pretty terminal output and text-report writing;
- selecting random frame pairs;
- validating CSV numbers;
- converting pose CSV rows into SpatialMath ``SE3`` objects;
- computing trusted epipolar geometry from KITTI ground truth.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spatialmath as sm


STATUS_STYLES = {
    "[PASS]": ("✅ PASS", "\033[32m"),
    "[FAIL]": ("❌ FAIL", "\033[31m"),
    "[INFO]": ("ℹ️  INFO", "\033[36m"),
    "[WARN]": ("⚠️  WARN", "\033[33m"),
}


def decorate_status_messages(text: str, use_color: bool) -> str:
    """Replace simple status tags such as ``[PASS]`` with icons and colour."""

    for plain_tag, (pretty_tag, colour) in STATUS_STYLES.items():
        if use_color:
            text = text.replace(plain_tag, f"{colour}{pretty_tag}\033[0m")
        else:
            text = text.replace(plain_tag, pretty_tag)
    return text


def remove_terminal_colours(text: str) -> str:
    """Remove ANSI colour codes before writing the plain text report."""

    return re.sub(r"\033\[[0-9;]*m", "", text)


class TeeOutput:
    """Write printed output to the terminal and optionally a report file."""

    def __init__(self, terminal, report_file, use_color: bool):
        self.terminal = terminal
        self.report_file = report_file
        self.use_color = use_color

    def write(self, text: str) -> int:
        terminal_text = decorate_status_messages(text, self.use_color)
        report_text = remove_terminal_colours(decorate_status_messages(text, False))

        self.terminal.write(terminal_text)
        if self.report_file is not None:
            self.report_file.write(report_text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        if self.report_file is not None:
            self.report_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


class FirstNFrames:
    """Small dataset wrapper used for quick local tests.

    The wrapped dataset behaves like the full KITTI dataset, except ``len()``
    reports only the first ``number_of_frames`` frames.
    """

    def __init__(self, dataset, number_of_frames: int):
        if number_of_frames < 2:
            raise ValueError("--max-frames must be at least 2")
        self.dataset = dataset
        self.number_of_frames = min(number_of_frames, len(dataset))

    def __len__(self):
        return self.number_of_frames

    @property
    def frame_count(self):
        return self.number_of_frames

    def stereo(self, index: int):
        return self.dataset.stereo(index)

    def camera_calibration(self, camera: int = 2):
        return self.dataset.camera_calibration(camera)

    def ground_truth_pose(self, index: int):
        return self.dataset.ground_truth_pose(index)


def is_number(value: str) -> bool:
    """Return True when ``value`` can be interpreted as a finite number."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def remove_old_file(filename: str) -> Path:
    """Remove a previous output file so this run cannot pass by accident."""

    path = Path.cwd() / filename
    if path.exists():
        path.unlink()
    return path


def choose_random_frame_pairs(
    number_of_frames: int,
    number_of_pairs: int,
    seed: int,
    skip_frames: int = 1,
) -> list[tuple[int, int]]:
    """Choose random frame pairs separated by ``skip_frames`` frames."""

    if number_of_frames < 2:
        raise ValueError("The dataset must contain at least two frames.")
    if skip_frames < 1:
        raise ValueError("skip_frames must be at least 1.")
    if number_of_frames <= skip_frames:
        raise ValueError(
            f"The dataset has {number_of_frames} frame(s), so it cannot provide "
            f"pairs separated by {skip_frames} frame(s)."
        )

    possible_start_frames = list(range(number_of_frames - skip_frames))
    random_generator = random.Random(seed)
    random_generator.shuffle(possible_start_frames)

    selected_start_frames = sorted(
        possible_start_frames[: min(number_of_pairs, len(possible_start_frames))]
    )
    return [(frame_i, frame_i + skip_frames) for frame_i in selected_start_frames]


def choose_random_consecutive_frame_pairs(
    number_of_frames: int,
    number_of_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Backward-compatible wrapper for adjacent-frame pairs."""

    return choose_random_frame_pairs(number_of_frames, number_of_pairs, seed, 1)


def skew(vector) -> np.ndarray:
    """Return the matrix form of a cross product with ``vector``."""

    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def left_camera_pose_from_ground_truth(full_dataset, frame: int) -> sm.SE3:
    """Return the ground-truth pose of the left camera for one KITTI frame."""

    ground_truth_imu_pose = full_dataset.ground_truth_pose(frame)
    T_camera_imu = sm.SE3(
        full_dataset.camera_calibration(camera=2)["T_cam_imu"],
        check=False,
    )
    return T_camera_imu @ ground_truth_imu_pose @ T_camera_imu.inv()


def trusted_fundamental_matrix(full_dataset, frame_i: int, frame_j: int) -> np.ndarray:
    """Compute the trusted fundamental matrix for a left-image pair.

    The returned matrix is ``F_ji`` and is used as:

    ``x_j.T @ F_ji @ x_i = 0``

    where ``x_i`` and ``x_j`` are corresponding image points in homogeneous
    pixel coordinates.
    """

    camera_pose_i = left_camera_pose_from_ground_truth(full_dataset, frame_i)
    camera_pose_j = left_camera_pose_from_ground_truth(full_dataset, frame_j)

    # Convert from two absolute left-camera poses to the transform that maps
    # 3D points from camera i coordinates into camera j coordinates.
    transform_i_to_j = camera_pose_j.inv() @ camera_pose_i
    R_i_to_j = transform_i_to_j.R
    t_i_to_j = transform_i_to_j.t

    essential_matrix = skew(t_i_to_j) @ R_i_to_j

    K = full_dataset.camera_calibration(camera=2)["K"].astype(float)
    K_inverse = np.linalg.inv(K)
    fundamental_matrix = K_inverse.T @ essential_matrix @ K_inverse

    # The scale of F is arbitrary. Normalising makes debugging output stable.
    norm = np.linalg.norm(fundamental_matrix)
    if norm > 0.0:
        fundamental_matrix = fundamental_matrix / norm
    return fundamental_matrix


def sampson_errors_pixels(rows, fundamental_matrix: np.ndarray) -> list[float]:
    """Calculate square-root Sampson errors for submitted feature matches."""

    errors = []
    for row in rows:
        point_i = np.array(
            [float(row["u_i"]), float(row["v_i"]), 1.0],
            dtype=float,
        )
        point_j = np.array(
            [float(row["u_j"]), float(row["v_j"]), 1.0],
            dtype=float,
        )

        F_x_i = fundamental_matrix @ point_i
        F_transpose_x_j = fundamental_matrix.T @ point_j
        epipolar_residual = float(point_j.T @ fundamental_matrix @ point_i)

        denominator = (
            F_x_i[0] ** 2
            + F_x_i[1] ** 2
            + F_transpose_x_j[0] ** 2
            + F_transpose_x_j[1] ** 2
        )
        if denominator <= 0.0:
            continue

        sampson_distance_squared = (epipolar_residual**2) / denominator
        errors.append(math.sqrt(sampson_distance_squared))
    return errors


def pose_from_csv_row(row, angles_are_degrees: bool = False) -> sm.SE3:
    """Build an SE3 pose from one relative-pose or trajectory CSV row."""

    roll = float(row["roll"])
    pitch = float(row["pitch"])
    yaw = float(row["yaw"])
    if angles_are_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    return sm.SE3.Trans(
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
    ) * sm.SE3.RPY(
        roll,
        pitch,
        yaw,
        order="zyx",
    )


def rotation_angle_rad(pose: sm.SE3) -> float:
    """Return the magnitude of the rotational part of an SE3 pose in radians."""

    trace_value = float(np.trace(pose.R))
    cosine_angle = (trace_value - 1.0) / 2.0
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(math.acos(cosine_angle))


def plot_trajectory_comparison(student_poses, ground_truth_poses, filename: str) -> None:
    """Plot estimated and ground-truth trajectories in the camera x/z plane."""

    student_positions = np.asarray([pose.t for pose in student_poses])
    ground_truth_positions = np.asarray([pose.t for pose in ground_truth_poses])

    plt.figure(figsize=(7, 5))
    plt.plot(
        ground_truth_positions[:, 0],
        ground_truth_positions[:, 2],
        "-",
        linewidth=2,
        label="KITTI ground truth",
    )
    plt.plot(
        student_positions[:, 0],
        student_positions[:, 2],
        "--",
        linewidth=2,
        label="submitted VO",
    )
    plt.scatter(
        ground_truth_positions[0, 0],
        ground_truth_positions[0, 2],
        c="green",
        s=80,
        label="start",
    )
    plt.scatter(
        ground_truth_positions[-1, 0],
        ground_truth_positions[-1, 2],
        c="red",
        s=80,
        label="end",
    )
    plt.xlabel("x: right [m]")
    plt.ylabel("z: forward [m]")
    plt.title("Visual odometry trajectory comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def warn_if_angles_look_like_degrees(
    label: str,
    rows,
    rotation_rmse_rad: float,
    rotation_rmse_if_degrees_rad: float,
) -> None:
    """Warn if the submitted roll/pitch/yaw values probably use degrees."""

    largest_angle_value = max(
        abs(float(row[column]))
        for row in rows
        for column in ["roll", "pitch", "yaw"]
    )

    if largest_angle_value > 2.0 * math.pi:
        print(
            f"[WARN] {label}: at least one angle is larger than 2*pi. "
            "The CSV should use radians, not degrees."
        )
        return

    degrees_interpretation_is_much_better = (
        rotation_rmse_rad > 0.01
        and rotation_rmse_if_degrees_rad < 0.3 * rotation_rmse_rad
    )
    if degrees_interpretation_is_much_better:
        print(
            f"[WARN] {label}: your roll/pitch/yaw values look like they may be "
            "in degrees. The required unit is radians."
        )
        print(
            f"       Rotation RMSE if read as radians: {rotation_rmse_rad:.6f} rad"
        )
        print(
            "       Rotation RMSE if read as degrees and converted to radians: "
            f"{rotation_rmse_if_degrees_rad:.6f} rad"
        )
