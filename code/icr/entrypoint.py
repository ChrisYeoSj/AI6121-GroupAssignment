from best_view import scores

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

def entrypoint():
    r = load("yunnan_rock_1")
    
    start = initialise_reconstruction(r["K"],r["kpts"],r["matches"],random_seed=0)

    poses = start.poses


    track_points = cr_track_points(r["tracks"], start.observations)
    
    icr_scores = scores(r["tracks"], track_points, poses)

