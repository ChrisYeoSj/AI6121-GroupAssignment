"""
Default openCV PnP implementation using defualt RANSAC parameters.
References: 
 - https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
 - Colmap reprojection error: https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.h
 - Look line 63 of their file
"""

import cv2
import numpy as np
def open_cv_pnp(points_2D, points_3D, K):
    return cv2.solvePnPRansac(points_3D, points_2D, K, None, 
    iterationsCount = 100, reprojectionError = 12.0, confidence = 0.99,flags=cv2.SOLVEPNP_EPNP)