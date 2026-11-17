from best_view import scores
from pnp import open_cv_pnp

# Thought Chris or Asmita would provide this so wrote it myself guys
def check_observation(track, camera_id, keypoint_id):
    for item in track:
        if item[0] == camera_id and item[1] == keypoint_id:
            return True
    return False

def loop_observations(track, i, observations):
    for cam_id, key_id in observations.items():
        matched = check_observation(track, cam_id, int(key_id[i]))
        if matched == False:
            return False
    return True

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

    print("Selected camera:", best_camera)
    print("Number of 2D-3D matches:", len(points_3D))
    print("PnP inliers:", len(pnp_results[3]))
    print("Inlier ratio:", ratio_inliners)
    print("Estimated R:", R)
    print("Estimated t:", t)
    
