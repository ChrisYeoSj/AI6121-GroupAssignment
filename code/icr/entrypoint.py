from best_view import scores

# Thought Chris or Asmita would provide this so wrote it myself guys
def check_observation(track, camera_id, keypoint_id):

    for item in track:
        if item[0] == camera_id and item[1] == keypoint_id:
            return True
    return False

def find_match_points(track, observations):
    match_points = []
    overvation_vals = list(observations.values())
    for i in range(len(overvation_vals[0])):
        mathed = None
        for c_id, k_id in observations.items():
            keypoint_id = int(k_id[i])

            matched = check_observation(track, c_id, keypoint_id)

            if matched == False:
                break
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


def main():
    tracks = {
    42: [(0, 10), (1, 15), (2, 20)],
    55: [(0, 30), (2, 40)]
}
    observations = {
    0: [10, 30],
    1: [15, 25]
}

    track_points = cr_track_points(tracks, observations)
    print("Track Points:", track_points)

main()

