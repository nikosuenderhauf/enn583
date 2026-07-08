"""Run the ENN583 student solution on a small KITTI example.

This file is intentionally written as a simple script rather than a clever test
framework. Read the three ``check_*`` functions below to see exactly how your
three functions in ``src/student_solution.py`` are called.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time

import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SUPPORT_DIR = REPO_ROOT / "support"

sys.path.insert(0, str(SUPPORT_DIR))
sys.path.insert(0, str(SRC_DIR))

import kitti_utils as kitti
import student_solution


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


# ====================================================================================
# Main Checking Functions, one for each of the three student functions. 
# 
# Each function calls the student function, checks that it ran without error, 
# and checks that the output CSV file exists and has the expected format.
# ====================================================================================
def check_match_features(img_i, img_j) -> bool:
    """Call ``match_features(img_i, img_j)`` and check ``results_matches.csv``."""

    print("\n" + "=" * 80)
    print("Checking match_features(img_i, img_j)")
    print("=" * 80)

    output_file = remove_old_file("results_matches.csv")

    start = time.perf_counter()
    try:
        student_solution.match_features(img_i, img_j)
        function_ran = True
        print("[PASS] match_features() ran without errors.")
    except Exception as exc:
        function_ran = False
        print("[FAIL] match_features() raised an error:")
        print(exc)
    runtime = time.perf_counter() - start
    print(f"[INFO] Runtime: {runtime:.3f} seconds")

    if not output_file.exists():
        print("[FAIL] match_features() must create results_matches.csv.")
        return False
    print("[PASS] Found results_matches.csv.")

    with output_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    expected_header = ["match_id", "u_i", "v_i", "u_j", "v_j"]
    if reader.fieldnames != expected_header:
        print("[FAIL] results_matches.csv must have this exact header:")
        print(",".join(expected_header))
        return False
    print("[PASS] results_matches.csv has the expected header.")

    if not rows:
        print("[FAIL] results_matches.csv must contain at least one match.")
        return False
    print(f"[PASS] results_matches.csv contains {len(rows)} match row(s).")

    rows_are_valid = True
    for expected_match_id, row in enumerate(rows):
        rows_are_valid &= row["match_id"] == str(expected_match_id)
        rows_are_valid &= is_number(row["u_i"])
        rows_are_valid &= is_number(row["v_i"])
        rows_are_valid &= is_number(row["u_j"])
        rows_are_valid &= is_number(row["v_j"])

    if rows_are_valid:
        print("[PASS] Match rows have sequential IDs and numeric coordinates.")
    else:
        print(
            "[FAIL] Match rows must use match_id values 0, 1, 2, ... and "
            "numeric u/v coordinates."
        )

    return function_ran and rows_are_valid

# ====================================================================================
def check_estimate_relative_pose(dataset, full_dataset, frame_i: int, frame_j: int) -> bool:
    """Call ``estimate_relative_pose(...)`` and check ``results_relative_pose.csv``."""

    print("\n" + "=" * 80)
    print("Checking estimate_relative_pose(dataset, frame_i, frame_j)")
    print("=" * 80)

    output_file = remove_old_file("results_relative_pose.csv")

    start = time.perf_counter()
    try:
        student_solution.estimate_relative_pose(dataset, frame_i, frame_j)
        function_ran = True
        print("[PASS] estimate_relative_pose() ran without errors.")
    except Exception as exc:
        function_ran = False
        print("[FAIL] estimate_relative_pose() raised an error:")
        print(exc)
    runtime = time.perf_counter() - start
    print(f"[INFO] Runtime: {runtime:.3f} seconds")

    if not output_file.exists():
        print("[FAIL] estimate_relative_pose() must create results_relative_pose.csv.")
        return False
    print("[PASS] Found results_relative_pose.csv.")

    with output_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    expected_header = ["frame_i", "frame_j", "x", "y", "z", "roll", "pitch", "yaw"]
    if reader.fieldnames != expected_header:
        print("[FAIL] results_relative_pose.csv must have this exact header:")
        print(",".join(expected_header))
        return False
    print("[PASS] results_relative_pose.csv has the expected header.")

    if len(rows) != 1:
        print("[FAIL] results_relative_pose.csv must contain exactly one data row.")
        return False
    row = rows[0]
    print("[PASS] results_relative_pose.csv contains one data row.")

    if row["frame_i"] != str(frame_i) or row["frame_j"] != str(frame_j):
        print(
            f"[FAIL] The row must describe frames {frame_i} -> {frame_j}, "
            f"but it describes {row['frame_i']} -> {row['frame_j']}."
        )
        return False

    numeric_columns = ["x", "y", "z", "roll", "pitch", "yaw"]
    if not all(is_number(row[column]) for column in numeric_columns):
        print("[FAIL] x, y, z, roll, pitch, and yaw must be numeric.")
        return False
    print("[PASS] Relative-pose row has numeric pose values.")

    # Convert the student's CSV row into an SE3 pose. Angles are in radians.
    student_pose = sm.SE3.Trans(
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
    ) * sm.SE3.RPY(
        float(row["roll"]),
        float(row["pitch"]),
        float(row["yaw"]),
        order="zyx",
    )

    # The full KITTI dataset has ground truth. The restricted dataset passed to
    # student code above does not.
    ground_truth_pose_i = full_dataset.ground_truth_pose(frame_i)
    ground_truth_pose_j = full_dataset.ground_truth_pose(frame_j)
    ground_truth_relative_pose = ground_truth_pose_i.inv() @ ground_truth_pose_j

    pose_error = ground_truth_relative_pose.inv() @ student_pose
    translation_rmse = math.sqrt(sum(value * value for value in pose_error.t) / 3)
    rotation_rmse_rad = math.sqrt(
        sum(value * value for value in pose_error.rpy(order="zyx")) / 3
    )
    rotation_rmse_deg = math.degrees(rotation_rmse_rad)

    print("[INFO] Relative-pose comparison against KITTI ground truth:")
    print(f"       Translation RMSE: {translation_rmse:.6f} m")
    print(
        f"       Rotation RMSE: {rotation_rmse_rad:.6f} rad "
        f"({rotation_rmse_deg:.3f} deg)"
    )

    return function_ran

# ====================================================================================
def check_visual_odometry(dataset) -> bool:
    """Call ``visual_odometry(dataset)`` and check ``results_visual_odometry.csv``."""

    print("\n" + "=" * 80)
    print("Checking visual_odometry(dataset)")
    print("=" * 80)

    output_file = remove_old_file("results_visual_odometry.csv")

    start = time.perf_counter()
    try:
        student_solution.visual_odometry(dataset)
        function_ran = True
        print("[PASS] visual_odometry() ran without errors.")
    except Exception as exc:
        function_ran = False
        print("[FAIL] visual_odometry() raised an error:")
        print(exc)
    runtime = time.perf_counter() - start
    print(f"[INFO] Runtime: {runtime:.3f} seconds")

    if not output_file.exists():
        print("[FAIL] visual_odometry() must create results_visual_odometry.csv.")
        return False
    print("[PASS] Found results_visual_odometry.csv.")

    with output_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    expected_header = ["frame", "x", "y", "z", "roll", "pitch", "yaw"]
    if reader.fieldnames != expected_header:
        print("[FAIL] results_visual_odometry.csv must have this exact header:")
        print(",".join(expected_header))
        return False
    print("[PASS] results_visual_odometry.csv has the expected header.")

    if not rows:
        print("[FAIL] results_visual_odometry.csv must contain at least one pose.")
        return False
    print(f"[PASS] results_visual_odometry.csv contains {len(rows)} pose row(s).")

    rows_are_valid = True
    for expected_frame, row in enumerate(rows):
        rows_are_valid &= row["frame"] == str(expected_frame)
        rows_are_valid &= is_number(row["x"])
        rows_are_valid &= is_number(row["y"])
        rows_are_valid &= is_number(row["z"])
        rows_are_valid &= is_number(row["roll"])
        rows_are_valid &= is_number(row["pitch"])
        rows_are_valid &= is_number(row["yaw"])

    if rows_are_valid:
        print("[PASS] Trajectory rows have sequential frame IDs and numeric poses.")
    else:
        print(
            "[FAIL] Trajectory rows must use frame values 0, 1, 2, ... and "
            "numeric x/y/z/roll/pitch/yaw values."
        )

    return function_ran and rows_are_valid



# ====================================================================================
# ====================================================================================
# ====================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check src/student_solution.py on a small KITTI example.",
    )
    parser.add_argument(
        "--sequence",
        default="2011_09_26_drive_0035",
        help="KITTI sequence name to load.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional data directory containing or receiving the KITTI data.",
    )
    parser.add_argument("--frame-i", type=int, default=0)
    parser.add_argument("--frame-j", type=int, default=1)
    args = parser.parse_args()

    if args.data_dir is not None:
        os.environ["ENN583_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())

    print("Loading KITTI data...")
    full_dataset = kitti.load_kitti_dataset(name=args.sequence)
    dataset = full_dataset.student_view(include_ground_truth=False)

    if not (0 <= args.frame_i < len(dataset)) or not (0 <= args.frame_j < len(dataset)):
        raise SystemExit(
            f"Frame indices must be between 0 and {len(dataset) - 1}; "
            f"got {args.frame_i} and {args.frame_j}."
        )

    img_i = dataset.stereo(args.frame_i)[0]
    img_j = dataset.stereo(args.frame_j)[0]

    print(f"Loaded sequence: {args.sequence}")
    print(f"Number of frames: {len(dataset)}")
    print(f"Using frames {args.frame_i} and {args.frame_j}")
    print(f"Image {args.frame_i} shape: {img_i.shape}")
    print(f"Image {args.frame_j} shape: {img_j.shape}")
    print("\nThis checker does not assign marks.")
    print("It shows whether your functions run and whether the CSV files look right.")

    match_ok = check_match_features(img_i, img_j)
    relative_pose_ok = check_estimate_relative_pose(
        dataset,
        full_dataset,
        args.frame_i,
        args.frame_j,
    )
    visual_odometry_ok = check_visual_odometry(dataset)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"match_features(): {'PASS' if match_ok else 'FAIL'}")
    print(f"estimate_relative_pose(): {'PASS' if relative_pose_ok else 'FAIL'}")
    print(f"visual_odometry(): {'PASS' if visual_odometry_ok else 'FAIL'}")

    if not (match_ok and relative_pose_ok and visual_odometry_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
