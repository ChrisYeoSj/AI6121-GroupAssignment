"""
Default openCV PnP implementation using defualt RANSAC parameters.
References: 
 - https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
 - Colmap reprojection error: https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.h
 - Look line 63 of their file
"""

import cv2
import numpy as np

# refine the ransac pose on its inliers, like colmap
REFINE_POSE = True

def open_cv_pnp(points_2D, points_3D, K):
    result = cv2.solvePnPRansac(points_3D, points_2D, K, None,
    iterationsCount = 100, reprojectionError = 12.0, confidence = 0.99,flags=cv2.SOLVEPNP_EPNP)
    success, rvec, tvec, inliers = result
    if not REFINE_POSE or not success or inliers is None or len(inliers) < 4:
        return result
    rows = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(points_3D[rows], points_2D[rows], K, None, rvec, tvec)
    return success, rvec, tvec, inliers