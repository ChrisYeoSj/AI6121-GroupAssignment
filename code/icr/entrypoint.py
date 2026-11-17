from best_view import scores
from pnp import open_cv_pnp
from triangulate import triangulates, check_positive_depth, reproj, resize_projection
import numpy as np
import cv2
# for testing
import pycolmap

# Chris / Asmita feature track with 3d point (as we agreed in word doc)
def check_observation(track, camera_id, keypoint_id):
    for item in track:
        if item[0] == camera_id and item[1] == keypoint_id:
            return True
    return False
# Chris / Asmita feature track with 3d point (as we agreed in word doc)
def loop_observations(track, i, observations):
    for tuple_item in observations.items():
        matched = check_observation(track, tuple_item[0], int(tuple_item[1][i]))
        if matched == False:
            return False
    return True

# Chris / Asmita feature track with 3d point (as we agreed in word doc)
def find_match_points(track, observations):
    match_points = []
    observation_vals = list(observations.values())
    for i in range(len(observation_vals[0])):
        matched = loop_observations(track, i, observations)
        if matched == True:
            match_points.append(i)
    return match_points

 
def cr_track_points(tracks, observations): 
    track_points = {}

    for track_id, track in tracks.items():
        match_points = find_match_points(track, observations)
        track_points[track_id] = match_points
    
    return track_points

def match_points_for_pnp(points3d, feature_tracks, track_points, keypoints, cam):
    points_2D, points_3D = list(), list()
    for track_id, track in feature_tracks.items():
        # Need to check if track can have more than one point in same camera:
        track_points_length = len(track_points.get(track_id, []))
        if track_points_length == 1:
            for tuple_item in track:
                camera = tuple_item[0]
                if camera == cam:
                    point_3d = points3d[track_points[track_id][0]]
                    point_2d = keypoints[camera][int(tuple_item[1])]
                    
                    points_3D.append(point_3d)
                    points_2D.append(point_2d)

    return (points_2D, points_3D)

def match_points_for_triangulation(feature_tracks, track_points, keypoints, cam1, cam2):
    
    points1, points2, relation_match = np.array([]), np.array([]), np.array([], dtype=int)

    for track_id, track in feature_tracks.items():
        track_points_length = len(track_points.get(track_id, []))
        if track_points_length == 0:
            # Using dict because data is becoming large and confusing to keep track
            camera_info = {cam1:{ "key": None, "count":0}, cam2: { "key": None, "count":0}}
            for tuple_item in track:
                camera_id = tuple_item[0]
                keypoint_id = tuple_item[1]
                if camera_id == cam1:
                    camera_info[cam1]["key"] = int(keypoint_id)
                    camera_info[cam1]["count"] += 1
                elif camera_id == cam2:
                    camera_info[cam2]["key"] = int(keypoint_id)
                    camera_info[cam2]["count"] += 1
            if camera_info[cam1]["count"] == 1 and camera_info[cam2]["count"] == 1:
                points1 = np.append(points1, keypoints[cam1][camera_info[cam1]["key"]])
                points2 = np.append(points2, keypoints[cam2][camera_info[cam2]["key"]])
                relation_match = np.append(relation_match, track_id)
    return (points1.reshape(-1, 2), points2.reshape(-1, 2), relation_match)

def entrypoint():
    r = load("yunnan_rock_1")
    
    start = initialise_reconstruction(r["K"],r["kpts"],r["matches"],random_seed=0)

    poses = start.poses

    track_points = cr_track_points(r["tracks"], start.observations)
    
    icr_scores = scores(r["tracks"], track_points, poses, r["kpts"])
    print("Simple scores:", icr_scores[0])
    print("Grid scores:", icr_scores[1])
    
    print("Registered cameras:", list(poses.keys()))
    print("Number of feature tracks:", len(r["tracks"]))
    
    # Comment to switch guys

    # For Simple score:
    # simple_scores = icr_scores[0]
    # best_camera = simple_scores[0]
    
    # For Grid score:
    grid_scores = icr_scores[1]
    if not grid_scores:
        print("No grid scores found")
        return
    best_camera = max(grid_scores, key=grid_scores.get)

    matched_points = match_points_for_pnp(start.points, r["tracks"], track_points, r["kpts"], best_camera)

    points_2D = matched_points[0]
    points_3D = matched_points[1]
    
    """
    Followed Colmap minmum inliers, ratio for PnP, reference file is below
    References:
    - https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.h
    - Line 66 for minmum inliers
    - Line 69 for ratio
    """
    if len(points_3D) < 30:
        print("Not enough points for PnP")
        return
    pnp_results = open_cv_pnp(np.asarray(points_2D, dtype=np.float64), np.asarray(points_3D, dtype=np.float64), np.asarray(r["K"], dtype=np.float64))

    if not pnp_results[0] or pnp_results[3] is None:
        print("PnP failed")
        return

    if len(pnp_results[3]) < 30:
        print("Not enough inliers for PnP")
        return
    ratio_inliners = len(pnp_results[3]) / len(points_3D)

    if ratio_inliners < 0.25:
        print("Ratio to low")
        return
    
    R, _ = cv2.Rodrigues(pnp_results[1])
    t = pnp_results[2].reshape((3))

    poses[best_camera] = (R, t)

    for cam in poses:
        if cam != best_camera: 
            print("Current pair", cam, best_camera)
            matched_points_from_triangulation = match_points_for_triangulation(r["tracks"], track_points, r["kpts"], cam, best_camera)

            print("Number of points for triangulation:", len(matched_points_from_triangulation[2]))

            if (len(matched_points_from_triangulation[2]) > 0):
                new_points_3D = triangulates(poses[cam][0], poses[best_camera][0], poses[cam][1], poses[best_camera][1], r["K"], np.transpose(matched_points_from_triangulation[0]), np.transpose(matched_points_from_triangulation[1]))
                print("Number of new points from triangulation:", len(new_points_3D))
                
                P1 = resize_projection(poses[cam][0], poses[cam][1], r["K"])
                P2 = resize_projection(poses[best_camera][0], poses[best_camera][1], r["K"])
                
                errors = reproj(P1, new_points_3D, matched_points_from_triangulation[0])
                errors2 = reproj(P2, new_points_3D, matched_points_from_triangulation[1])

                valid_points_trackid = {}

                for i in range(len(new_points_3D)):
                    track_id = int(matched_points_from_triangulation[2][i])

                    depth1 = check_positive_depth(poses[cam][0], poses[cam][1], new_points_3D[i])
                    depth2 = check_positive_depth(poses[best_camera][0], poses[best_camera][1], new_points_3D[i])

                    """
                    Colmap uses reprojection error of 4.0 pixels, reference file is below line 102
                    References: https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.h
                    """
                    if depth1 and depth2 and errors[0][i] < 4.0 and errors2[0][i] < 4.0:
                        valid_points_trackid[track_id] = new_points_3D[i]    
                
                print("Points passing depth validation and reprojection error validation:", len(valid_points_trackid))

                for tuple_item in valid_points_trackid.items():
                    track_id = tuple_item[0]
                    point_3D = tuple_item[1]
                    
                    start.points = np.append(start.points, [point_3D], axis=0)
                    track_points[track_id] = [len(start.points ) -1]

            else:
                print("No points to triangulate")
    print("Final number of points:", len(start.points))