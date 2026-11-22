import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import cv2

def collect_observations(poses, feature_tracks, track_points, keypoints):
    cam_idx, point_idx, observations = np.array([], dtype=int), np.array([], dtype=int), []
    camera_ids = list(poses.keys())
    for tuple_item in feature_tracks.items():
        track_id = tuple_item[0]
        track = tuple_item[1]
        track_points_length = len(track_points.get(track_id, []))
        if track_points_length == 1:
            point_index = track_points[track_id][0]
            for tuple_two_item in track:
                camera_id = tuple_two_item[0]
                keypoint_id = tuple_two_item[1]
                if camera_id in poses:
                    cam_idx = np.append(cam_idx, camera_ids.index(camera_id))
                    point_idx = np.append(point_idx, point_index)
                    observations.append(keypoints[camera_id][int(keypoint_id)])
    return (cam_idx, point_idx, np.array(observations).reshape(-1, 2), camera_ids)

"""
References"
https://computervisiononline.com/explains/bundle-adjustment/

Implementation of bundle adjustment follows the referenced bundle ajustment code also written in python (scroll at the bottom of reference page).
Minor modifications were made to the code to fit into our SFM Pipeline
- Cam Intrinsics are passed explicitly
- Intial camera poses are fixed.
-Soft-l1 is also used to reduce noise from observations and incorrect correspondences
"""

# Function can be found in the reference page, slight modifcations
def project(cams, X, K):
    Xc = Rotation.from_rotvec(cams[:, :3]).apply(X) + cams[:, 3:]
    projected = np.dot(Xc, np.transpose(K))
    return projected[:, :2] / projected[:, 2:3]
# Function can be found in the reference page, slight modifcations
def unpack(p, full, free, n_cams, n_points):
    full = full.copy()
    full[free] = p
    return full[:6 * n_cams].reshape((n_cams, 6)), full[6 * n_cams:].reshape((n_points, 3))
# Function can be found in the reference page, slight modifcations
def residuals(p, full, free, n_cams, n_points, cam_idx, point_idx, observations, K):
    cams, X = unpack(p, full, free, n_cams, n_points)
    return (project(cams[cam_idx], X[point_idx], K) - observations).ravel()
# Function can be found in the reference page, slight modifcations
def sparsity(n_cams, n_pts, cam_idx, point_idx, free):
    A = lil_matrix((2 * len(cam_idx), 6 * n_cams + 3 * n_pts), dtype=int)
    for k in range(len(cam_idx)):
        i = cam_idx[k]
        j = point_idx[k]
        A[2 * k:2 * k + 2, 6 * i:6 * i + 6] = 1
        A[2 * k:2 * k + 2, 6 * n_cams + 3 * j:6 * n_cams + 3 * j + 3] = 1
    return A.tocsc()[:, free]
# Function can be found in the reference page, slight modifcations
def lsq(poses, points_3d, cam_idx, point_idx, observations, K,
        variable_camera_ids=None, variable_point_indices=None, return_info=False,
        max_nfev=100):
    """Optional variable sets restrict BA; the first two cameras always stay fixed."""
    if not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    n_cams = len(poses)
    n_points = len(points_3d)

    camera_ids = list(poses.keys())
    cams0 = np.array([])

    for tuple_item in poses.items():
        camera_id = tuple_item[0]
        pose = tuple_item[1]
        R = pose[0]
        T = pose[1]
        rvec, _ = cv2.Rodrigues(R)
        cams0 = np.append(cams0, np.concatenate((rvec.flatten(), T.flatten())))
    cams0 = cams0.reshape((n_cams, 6))
    X0 = np.array(points_3d, dtype=float)
    full0 = np.r_[cams0.ravel(), X0.ravel()]

    free = np.ones(6 * n_cams + 3 * n_points, dtype=bool)
    free[:6] = False 
    free[:12] = False

    if variable_camera_ids is not None:
        variable_camera_ids = set(variable_camera_ids)
        for i, camera_id in enumerate(camera_ids):
            if camera_id not in variable_camera_ids:
                free[6 * i:6 * i + 6] = False
    if variable_point_indices is not None:
        free[6 * n_cams:] = False
        for point_index in variable_point_indices:
            free[6 * n_cams + 3 * point_index:6 * n_cams + 3 * point_index + 3] = True

    if not len(observations) or not free.any():
        raise ValueError("BA needs observations and at least one variable.")

    p0 = full0[free] 

    A = sparsity(n_cams, n_points, cam_idx, point_idx, free)

    result = least_squares(residuals, p0, jac_sparsity=A, x_scale='jac', loss='soft_l1', f_scale=4.0, max_nfev=max_nfev, args=(full0, free, n_cams, n_points, cam_idx, point_idx, observations, K))

    unpacked = unpack(result.x, full0, free, n_cams, n_points)

    for i, camera_id in enumerate(camera_ids):
        if not free[6 * i:6 * i + 6].any():
            continue
        rvec = unpacked[0][i, :3]
        tvec = unpacked[0][i, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        poses[camera_id] = (R, tvec)

    if return_info:
        info = dict(success=bool(result.success), status=int(result.status),
                    message=str(result.message), nfev=int(result.nfev), max_nfev=int(max_nfev),
                    cost=float(result.cost), optimality=float(result.optimality),
                    variable_cameras=[int(c) for i, c in enumerate(camera_ids)
                                      if free[6 * i:6 * i + 6].any()],
                    variable_points=int(free[6 * n_cams:].sum() // 3),
                    variables=int(free.sum()))
        return poses, unpacked[1], info
    return poses, unpacked[1]
