"""Public result-format checks shared by local and Gradescope grading."""

from __future__ import annotations

import spatialmath as sm


def check_results(results: object) -> dict[str, bool]:
    """Return checks for the required visual odometry result structure."""

    is_dictionary = type(results) is dict
    has_trajectory = is_dictionary and "trajectory" in results
    trajectory_is_list = has_trajectory and type(results["trajectory"]) is list
    trajectory_is_nonempty = trajectory_is_list and len(results["trajectory"]) > 0
    trajectory_contains_se3 = trajectory_is_nonempty and all(
        isinstance(pose, sm.SE3) for pose in results["trajectory"]
    )

    return {
        "is_dictionary": is_dictionary,
        "has_trajectory": has_trajectory,
        "trajectory_is_list": trajectory_is_list,
        "trajectory_is_nonempty": trajectory_is_nonempty,
        "trajectory_contains_se3": trajectory_contains_se3,
    }


def print_results_feedback(results: object) -> bool:
    """Print student-facing feedback and return whether all checks passed."""

    checks = check_results(results)

    if checks["is_dictionary"]:
        print("[PASS] Results is a dictionary.")
    else:
        print("[FAIL] The returned results should be a dictionary.")

    trajectory_list_valid = checks["has_trajectory"] and checks["trajectory_is_list"]
    if trajectory_list_valid:
        print("[PASS] Results contains a trajectory list.")
    else:
        print("[FAIL] Results must contain a 'trajectory' field holding a list.")

    trajectory_valid = (
        checks["trajectory_is_nonempty"] and checks["trajectory_contains_se3"]
    )
    if trajectory_valid:
        print("[PASS] The trajectory is a non-empty list of SE3 objects.")
    else:
        print("[FAIL] The trajectory must be non-empty and contain only SE3 objects.")

    return checks["is_dictionary"] and trajectory_list_valid and trajectory_valid

