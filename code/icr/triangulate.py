""" Triangulation implementation using OpenCV.
    References:
    - https://docs.opencv.org/4.13.0/d0/dbd/gr`oup__triangulation.html
"""
import numpy as np
import cv2


"""
Code: Computer Vision Online, "Reprojection Error"
https://computervisiononline.com/explains/reprojection-error/
This code (def project and def reproj) is available under the article above (scroll down to the code section). 
It is used to compute the reprojection error of 3D points given a camera projection matrix and 2D image points.
"""
def project(P, X):
    x = np.c_[X, np.ones(len(X))] @ P.T
    return x[:, :2] / x[:, 2:]

def reproj(P, X, x):
    e = np.linalg.norm(x - project(P, X), axis=1)
    return e, np.sqrt(np.mean(e ** 2))

"""
Fix Depth of 3D points to be positive in front of the camera.
References:
https://www.nature.com/articles/s41598-023-49553-9
As said in this article: 
In the area of 3D computer vision, the ability to estimate pose between two cameras under high noise levels 
while maintaining small reprojection errors reflects the robustness of such pose estimation algorithms. 
Moreover, maintaining positive depth constraint is another challenging task.
Unfortunately, current pose estimation algorithms are often sensitive to noise/outliers and do not always guarantee positive depths. 
As a standalone task, these algorithms perform a positive sign check and simply discard the points with negative depths after the algorithms are executed. 
These algorithms do not integrate positive depth constraints into the algorithms themselves. 

"""
def check_positive_depth(R, T, point_3d):
    camera_point = np.dot(R, point_3d) + T
    return camera_point[2] > 0

def resize_projection(R, T, K):
    return np.dot(K, np.column_stack(((R, T))))

def triangulate_points(R1, R2, T1, T2, K, points1, points2):
    triangulate = cv2.triangulatePoints(resize_projection(R1, T1, K), resize_projection(R2, T2, K), points1, points2)
    return triangulate

def triangulates(R1, R2, T1, T2, K, points1, points2):
    def get_3d_coordinate(curr, j, triangulate):
        return triangulate[j,curr] / triangulate[3,curr]
    
    triangulate = triangulate_points(R1, R2, T1, T2, K, points1, points2)
    points_3d = []

    for i in range(triangulate.shape[1]):
        points = []
        for j in range(3):
            points.append(get_3d_coordinate(i, j, triangulate))
        points_3d.append(points)
    return np.array(points_3d)


