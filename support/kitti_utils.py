"""Instructor-provided utilities for downloading and reading KITTI data.

The helpers in this module are intentionally lightweight so they work from both
scripts in ``src`` and weekly notebooks launched from different folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
import re
import numpy as np
import spatialmath as sm
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np


DEFAULT_DATE = "2011_09_26"
DEFAULT_DRIVE = "0035"
KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
KITTI_SEQUENCE_RE = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})(?:_drive_)(?P<drive>\d{4})(?:_sync)?$"
)


def parse_sequence_name(name: str) -> tuple[str, str]:
    """Parse a KITTI sequence name into ``(date, drive)``.

    Accepted forms are ``2011_09_26_drive_0013`` and
    ``2011_09_26_drive_0013_sync``.
    """

    match = KITTI_SEQUENCE_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "KITTI sequence names must look like "
            "'2011_09_26_drive_0013' or '2011_09_26_drive_0013_sync'."
        )
    return match.group("date"), match.group("drive")


def load_kitti_dataset(
    date: str = DEFAULT_DATE,
    drive: str = DEFAULT_DRIVE,
    data_dir: str | Path | None = None,
    *,
    name: str | None = None,
    download: bool = True,
    overwrite: bool = False,
) -> "KittiDataset":
    """Return a local KITTI dataset reader, downloading data if requested.

    Parameters
    ----------
    date, drive:
        KITTI sequence identifiers. You can also pass a full sequence name as
        the first argument, e.g. ``load_kitti_dataset("2011_09_26_drive_0013")``.
    name:
        Optional full sequence name, e.g. ``2011_09_26_drive_0013`` or
        ``2011_09_26_drive_0013_sync``.
    data_dir:
        Optional ENN583 data root. If omitted, data is stored in
        ``<repo>/data/kitti`` unless ``ENN583_DATA_DIR`` is set.
    download:
        If true, missing zip files are downloaded from the official KITTI S3
        mirror and extracted. If false, a ``FileNotFoundError`` is raised when
        the sequence is missing.
    overwrite:
        Re-download and re-extract the zip files even if they already exist.
    """

    if name is not None:
        date, drive = parse_sequence_name(name)
    elif "_drive_" in date:
        date, drive = parse_sequence_name(date)

    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve() / "kitti"
    elif os.environ.get("ENN583_DATA_DIR"):
        root = Path(os.environ["ENN583_DATA_DIR"]).expanduser().resolve() / "kitti"
    else:
        root = Path(__file__).resolve().parents[1] / "data" / "kitti"

    sequence = f"{date}_drive_{drive}_sync"
    drive_dir = root / date / sequence
    calibration_file = root / date / "calib_cam_to_cam.txt"
    dataset = KittiDataset(root=root, date=date, drive=drive)

    if overwrite or not (drive_dir.exists() and calibration_file.exists()):
        if not download:
            raise FileNotFoundError(
                f"KITTI drive {drive_dir} is missing. "
                "Call load_kitti_dataset(download=True) to download it."
            )
        root.mkdir(parents=True, exist_ok=True)
        downloads = [
            (
                f"{KITTI_BASE_URL}/{date}_drive_{drive}/{sequence}.zip",
                root / f"{sequence}.zip",
            ),
            (
                f"{KITTI_BASE_URL}/{date}_calib.zip",
                root / f"{date}_calib.zip",
            ),
        ]

        for url, zip_path in downloads:
            if overwrite or not zip_path.exists():
                print(f"Downloading {url}")
                urlretrieve(url, zip_path)
            with ZipFile(zip_path) as archive:
                archive.extractall(root)

    if not (drive_dir.exists() and calibration_file.exists()):
        raise FileNotFoundError(f"KITTI drive was not found after download: {drive_dir}")

    return dataset


@dataclass(frozen=True)
class KittiDataset:
    """Instructor view of one KITTI synced drive, including ground truth."""

    root: Path
    date: str = DEFAULT_DATE
    drive: str = DEFAULT_DRIVE

    @cached_property
    def data(self):
        """Return the underlying ``pykitti.raw`` dataset object."""

        try:
            import pykitti
        except ImportError as exc:
            raise ImportError(
                "pykitti is required to load KITTI data. Install the ENN583 "
                "environment from environment.yml, or run `pip install pykitti`."
            ) from exc

        return pykitti.raw(str(self.root), self.date, self.drive)

    def __len__(self) -> int:
        return self.frame_count

    def student_view(self, *, include_ground_truth: bool = False):
        """Return the dataset interface passed to student code."""

        if include_ground_truth:
            return self
        calibrations = {
            camera: {
                key: value.copy()
                for key, value in self.camera_calibration(camera).items()
            }
            for camera in range(4)
        }
        return KittiSequence(
            left_image_paths=self.data.cam2_files,
            right_image_paths=self.data.cam3_files,
            calibrations=calibrations,
        )

    def stereo(self, index: int):
        """Return the left and right RGB camera images at the specified n-th pose.

        The returned images are NumPy arrays from camera 2 and camera 3.
        """
        left = np.asarray(self.data.get_cam2(index).convert("RGB"))
        right = np.asarray(self.data.get_cam3(index).convert("RGB"))
        return left, right

    def ground_truth_pose(self, index: int):
        """Return the ground truth pose at the n-th frame as a 4x4 homogeneous transformation matrix.
          Notice that the pose is returned relative to the first frame, so the first pose is always the identity matrix."""

        # We want to return the pose if the n-th frame relative to the first frame, so we need to invert the first pose and apply it to all subsequent poses.
        T_w_imu0 = sm.SE3(self.data.oxts[0].T_w_imu)

        T_w_imu = sm.SE3(self.data.oxts[index].T_w_imu)

        # The pose of the n-th frame relative to the first frame is given by T_w_imu0.inv() @ T_w_imu, where T_w_imu0.inv() is the inverse of the first pose and T_w_imu is the pose of the n-th frame. This gives us the transformation from the first frame to the n-th frame, which is what we want to return.
        return T_w_imu0.inv() @ T_w_imu


        # return self.data.oxts[index].T_w_imu

    def camera_calibration(self, camera: int = 2) -> dict[str, object]:
        """Return calibration matrices for a camera.

        ``K`` is the 3x3 intrinsic matrix, ``P`` is the 3x4 rectified projection
        matrix, and ``T_cam_imu`` transforms points from the IMU frame into the
        camera frame.
        """

        if camera not in (0, 1, 2, 3):
            raise ValueError("camera must be 0, 1, 2, or 3")

        calib = self.data.calib
        return {
            "K": getattr(calib, f"K_cam{camera}"),
            "intrinsics": getattr(calib, f"K_cam{camera}"),
            "P": getattr(calib, f"P_rect_{camera}0"),
            "T_cam_imu": getattr(calib, f"T_cam{camera}_imu"),
        }

    @property
    def frame_count(self) -> int:
        """Return the number of poses/images in this sequence."""

        return len(self.data.timestamps)


class KittiSequence:
    """Restricted KITTI interface for student visual odometry code.

    This view exposes stereo images and camera calibration, but not ground-truth
    poses or the underlying pykitti object.
    """

    __slots__ = ("__left_image_paths", "__right_image_paths", "__calibrations")

    def __init__(
        self,
        left_image_paths: list[str],
        right_image_paths: list[str],
        calibrations: dict[int, dict[str, object]],
    ):
        self.__left_image_paths = tuple(left_image_paths)
        self.__right_image_paths = tuple(right_image_paths)
        self.__calibrations = calibrations

    def __len__(self) -> int:
        return self.frame_count

    def stereo(self, index: int):
        """Return the left and right RGB images at frame ``index``."""

        from PIL import Image

        with Image.open(self.__left_image_paths[index]) as image:
            left = np.asarray(image.convert("RGB")).copy()
        with Image.open(self.__right_image_paths[index]) as image:
            right = np.asarray(image.convert("RGB")).copy()
        return left, right

    def camera_calibration(self, camera: int = 2) -> dict[str, object]:
        """Return intrinsic, projection, and camera-to-IMU matrices."""

        if camera not in self.__calibrations:
            raise ValueError("camera must be 0, 1, 2, or 3")
        return {
            key: value.copy()
            for key, value in self.__calibrations[camera].items()
        }

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the sequence."""

        return min(
            len(self.__left_image_paths),
            len(self.__right_image_paths),
        )
