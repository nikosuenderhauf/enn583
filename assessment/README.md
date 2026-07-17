# Visual Odometry Assessment

Implement the three scaffolded functions in `src/student_solution.py`:

```python
def match_features(img_i, img_j):
    ...

def estimate_relative_pose(dataset, frame_i, frame_j):
    ...

def visual_odometry(dataset):
    ...
```

The `src/` directory is the student workspace. Students may add any supporting
Python modules or package subdirectories they need there. Instructor-provided
utilities, including `kitti_utils`, live outside `src/` and are installed by the
course environment.

## Run Locally

From the repository root:

```bash
python assessment/check_student_solution.py \
    --sequence 2011_09_26_drive_0035 \
    --data-dir data
```

The checker also writes the same output to
`check_student_solution_report.txt`, which can be useful when asking for help.
You can choose a different filename or disable the report:

```bash
python assessment/check_student_solution.py \
    --data-dir data \
    --report-file my_check_report.txt

python assessment/check_student_solution.py \
    --data-dir data \
    --no-report-file
```

By default, the checker runs all three subtasks. To run only one check, or a
subset of checks, use:

```bash
python assessment/check_student_solution.py \
    --data-dir data \
    --check-matches

python assessment/check_student_solution.py \
    --data-dir data \
    --check-matches \
    --check-relative-pose

python assessment/check_student_solution.py \
    --data-dir data \
    --check-visual-odometry
```

The command loads a real KITTI sequence and calls the three scaffolded
functions one after the other:

- `match_features(img_i, img_j)` receives left-image pairs from several
  randomly selected frame pairs.
- `estimate_relative_pose(dataset, frame_i, frame_j)` receives the dataset and
  several randomly selected frame pairs.
- `visual_odometry(dataset)` receives the dataset sequence.

It does not compare your output to a reference solution. For each subtask, it
checks that the function runs without errors, reports how long it took, creates
the required file, and writes CSV data in the required basic format.
For `match_features()`, the local checker also computes a trusted fundamental
matrix from KITTI calibration and ground-truth left-camera poses, then reports
the Sampson error of the submitted matches. This is feedback only; it is not a
mark.
For `estimate_relative_pose()`, the local checker also uses KITTI ground truth
to report translation and rotation RMSE averaged over those frame pairs. This
is feedback only; it is not a mark.

By default, the checker tests 10 random adjacent frame pairs using random seed 0
for the feature-matching and relative-pose checks. You can change the number of
pairs, the random seed, and the frame gap. For example,
`--skip-frames 5` checks pairs of the form `i -> i+5`:

```bash
python assessment/check_student_solution.py \
    --sequence 2011_09_26_drive_0035 \
    --data-dir data \
    --frame-pairs 5 \
    --skip-frames 5 \
    --random-seed 123
```

Required output files:

```csv
results_matches.csv
match_id,u_i,v_i,u_j,v_j
```

```csv
results_relative_pose.csv
frame_i,frame_j,x,y,z,roll,pitch,yaw
```

For `results_relative_pose.csv`, `x,y,z` are metres and
`roll,pitch,yaw` are radians. They should describe the left-camera pose of
`frame_j` relative to the left-camera pose of `frame_i`. The checker interprets
the angles using SpatialMath's
`SE3.RPY(roll, pitch, yaw, order="zyx")` convention.
If your angles look more consistent with degrees than radians, the checker will
print a warning.

```csv
results_visual_odometry.csv
frame,x,y,z,roll,pitch,yaw
```

The functions are not required to return anything. Later, hidden checks will
compare the contents of these files against reference results and award marks.

The command exits with status `0` when all public checks pass and status `1`
when a check fails.

When `visual_odometry()` is checked, the script also writes:

```text
results_visual_odometry_comparison.png
```

This plot shows your estimated trajectory and the KITTI ground-truth trajectory
in the top-down `x/z` plane. It is feedback only.

`check_student_solution.py` passes a restricted dataset object into the
dataset-based functions. It provides stereo frames, camera calibration, and
sequence length, but not ground-truth poses.

Most of the readable checking logic is in `check_student_solution.py`. The
support file `checker_utils.py` contains helper code for formatting output,
loading frame pairs, converting poses, and computing feedback-only geometry
metrics. You do not need to edit it.

Instructor note: the checker can also run another module with the same three
entry functions, for example:

```bash
python assessment/check_student_solution.py \
    --solution-module instructor.reference_solution_good \
    --max-frames 3
```
