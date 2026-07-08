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
src/student_solution.py
```

It must complete these three provided functions:

```python
def match_features(img_i, img_j):
    ...

def estimate_relative_pose(dataset, frame_i, frame_j):
    ...

def visual_odometry(dataset):
    ...
```

Do not change these function names or parameters. The runner loads KITTI,
passes test inputs into each function, and checks that each function creates
its required output file.

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
python assessment/check_student_solution.py \
    --sequence 2011_09_26_drive_0035 \
    --data-dir data
```

The checker loads a real KITTI sequence, selects two image frames, and gives
separate feedback for each subtask. It reports whether each function runs, how
long it took, and whether its CSV output has the expected basic format:

- `match_features(img_i, img_j)` runs without errors and creates
  `results_matches.csv` with header `match_id,u_i,v_i,u_j,v_j`;
- `estimate_relative_pose(dataset, frame_i, frame_j)` runs without errors and
  creates `results_relative_pose.csv` with header
  `frame_i,frame_j,x,y,z,roll,pitch,yaw`;
- `visual_odometry(dataset)` runs without errors and creates
  `results_visual_odometry.csv` with header `frame,x,y,z,roll,pitch,yaw`.

These checks are currently feedback-only. Later, Gradescope will compare the
output files against reference results and assign marks for correctness.
During strict marking, the dataset object does not expose ground-truth poses.
The local checker uses KITTI ground truth to report feedback-only translation
and rotation RMSE for `results_relative_pose.csv`.

## Gradescope Submission

Upload a ZIP containing the files and subdirectories **inside** `src/`.
Supporting modules and package subdirectories are allowed.

Required ZIP layout:

```text
student_solution.py
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

When the ZIP is opened, `student_solution.py` must be visible immediately at its
root. Do not include `src/` or another outer directory in the ZIP. You do not
need to submit datasets, notebooks, the environment, `assessment/`, or
`support/`.
