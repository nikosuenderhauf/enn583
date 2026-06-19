"""Local compatibility wrapper for the Gradescope result checks."""

from result_checks import print_results_feedback


class Autograder:
    """Print the same result-format feedback used by Gradescope."""

    def __init__(self, results):
        self.passed = print_results_feedback(results)
