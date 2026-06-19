# Visual Odometry Assessment

Implement `visual_odometry()` in `src/visual_odometry.py`.

The `src/` directory is the student workspace. Students may add any supporting
Python modules or package subdirectories they need there. Instructor-provided
utilities, including `kitti_utils`, live outside `src/` and are installed by the
course environment.

## Run Locally

From the repository root:

```bash
python assessment/run_vo.py \
    --sequence 2011_09_26_drive_0035 \
    --data-dir data
```

The command runs the implementation and checks the same public result-format
contract used by Gradescope:

- The returned value is a dictionary.
- It contains a `trajectory` list.
- The trajectory is non-empty.
- Every trajectory element is a `spatialmath.SE3` object.

The command exits with status `0` when all public checks pass and status `1`
when a check fails. Gradescope also runs private correctness and accuracy tests
that are not included in the student repository.

`run_vo.py` loads the selected sequence and passes a restricted dataset object
to `visual_odometry(dataset)`. It provides stereo frames, camera calibration,
and sequence length, but not ground-truth poses.
