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

The command loads a real KITTI sequence, selects two frames, and calls the
three scaffolded functions one after the other:

- `match_features(img_i, img_j)` receives the left images for frames 0 and 1.
- `estimate_relative_pose(dataset, frame_i, frame_j)` receives the dataset and
  the frame indices 0 and 1.
- `visual_odometry(dataset)` receives the dataset sequence.

It does not compare your output to a reference solution. For each subtask, it
checks that the function runs without errors, reports how long it took, creates
the required file, and writes CSV data in the required basic format.
For `estimate_relative_pose()`, the local checker also uses KITTI ground truth
to report translation and rotation RMSE for the pose in
`results_relative_pose.csv`. This is feedback only; it is not a mark.

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
`roll,pitch,yaw` are radians. The checker interprets the angles using
SpatialMath's `SE3.RPY(roll, pitch, yaw, order="zyx")` convention.

```csv
results_visual_odometry.csv
frame,x,y,z,roll,pitch,yaw
```

The functions are not required to return anything. Later, hidden checks will
compare the contents of these files against reference results and award marks.

The command exits with status `0` when all public checks pass and status `1`
when a check fails.

`check_student_solution.py` passes a restricted dataset object into the
dataset-based functions. It provides stereo frames, camera calibration, and
sequence length, but not ground-truth poses.
