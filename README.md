# ENN583 - Foundations of Robotic Vision

This repository contains the practical material and visual odometry assessment
scaffold for ENN583 taught at Queensland University of Technology (QUT) in Brisbane.

## Setup

Create and activate the course environment:

```bash
mamba env create --file environment.yml
conda activate enn583
pip install -e .
```

The editable install provides the instructor-supplied `kitti_utils` module used
by the practicals and assessment.

## Repository Layout

```text
src/                 Your visual odometry implementation
assessment/          Local assessment runner and public checks
practicals/          Weekly practical notebooks
support/             Instructor-provided support modules
data/                Downloaded datasets; not committed to git
```

The `src/` directory is your workspace. You may add Python files and package
subdirectories as needed, but the required entry point must remain:

```text
src/visual_odometry.py
```

It must define:

```python
def visual_odometry(dataset):
    ...
```

Do not change this function's name or parameters. The runner loads KITTI and
passes a prepared dataset object into your function.

## KITTI Data

The installed `kitti_utils` module downloads KITTI sequences into
`data/kitti` by default:

```python
from kitti_utils import load_kitti_dataset

dataset = load_kitti_dataset("2011_09_26_drive_0035")
left_rgb, right_rgb = dataset.stereo(0)
pose = dataset.ground_truth_pose(0)
calibration = dataset.camera_calibration(camera=2)
```

Set `ENN583_DATA_DIR` to use another data location:

```bash
export ENN583_DATA_DIR=/path/to/shared/data
```

## Run The Assessment Locally

From the repository root:

```bash
python assessment/run_vo.py \
    --sequence 2011_09_26_drive_0035 \
    --data-dir data
```

The public checks verify that:

- the returned value is a dictionary;
- it contains a `trajectory` list;
- the trajectory is non-empty;
- every trajectory element is a `spatialmath.SE3` object.

Gradescope also runs private tests that are not included in the student
repository. During strict marking, the dataset object does not expose
ground-truth poses.

## Gradescope Submission

Upload a ZIP containing the files and subdirectories **inside** `src/`.
Supporting modules and package subdirectories are allowed.

Required ZIP layout:

```text
visual_odometry.py
another_module.py
optional_package/
├── __init__.py
└── helpers.py
```

Create the ZIP from the repository root:

```bash
(cd src && zip -r ../enn583-submission.zip . \
    -x "*/__pycache__/" "*/__pycache__/*" "*.pyc")
```

When the ZIP is opened, `visual_odometry.py` must be visible immediately at its
root. Do not include `src/` or another outer directory in the ZIP. You do not
need to submit datasets, notebooks, the environment, `assessment/`, or
`support/`.
