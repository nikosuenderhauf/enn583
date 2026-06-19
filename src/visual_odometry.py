"""Student implementation of visual odometry for ENN583.

You may add helper functions, classes, and additional modules inside ``src/``.
Do not change the name or parameters of ``visual_odometry`` because the local
runner and Gradescope call that function directly.
"""

import numpy as np
import spatialmath as sm
import cv2


# ====================================================================================
# Your code goes here! 
# ====================================================================================
# Add your helper functions and classes below this line.
#
# You may use packages provided by environment.yml. Ask the teaching team before
# adding another dependency because it may not be installed in Gradescope.








# ====================================================================================
# ====================================================================================
# ====================================================================================
# Implement your visual odometry algorithm in this function. 
# You may call helper functions and classes that you define above.
# 
# DO NOT change the name, parameters, or return type of this function because the Gradescope autograder calls it directly.
def visual_odometry(dataset):
    """Estimate the camera trajectory for a prepared KITTI sequence.

    Parameters
    ----------
    dataset:
        Dataset supplied by the runner. It provides ``stereo(i)``,
        ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
        While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
        However, ground-truth poses are unavailable during marking.
    Returns
    -------
    dict
        A dictionary containing ``trajectory``. The trajectory must be a
        non-empty list of ``spatialmath.SE3`` poses. Pose ``i`` must describe
        frame ``i`` relative to frame 0, so the first pose should be identity.
    """

    # You will return the results of your implementation in this dictionary.
    # DO NOT rename or remove the fields in this dictionary, but you may add additional fields if you like.
    results = {
        "trajectory": [] # This should be a list of sm.SE3 poses, one for each frame, describing the camera pose relative to frame 0.    
            }


    # ====================================================================================
    # Your code goes here! 
    # ====================================================================================


    # DO NOT change the return type or dictionary structure.
    return results
