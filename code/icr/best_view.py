# Data ill get from Asmita and Chris (please update it if made mistakes)
# https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf
# Follows Section 4.2 as reference

def score_using_count(feature_tracks, track_points, poses):
    scores = {} 
    def score_for_single_3d_point(camera_id):
        if camera_id in poses:
            return 0
        else:
            return 1

    for track_id, points_in_track in feature_tracks.items():
        if track_points.get(track_id):
            for tuple_item in points_in_track:
                camera_id = tuple_item[0]
                score = score_for_single_3d_point(camera_id)
                if score == 1:
                    scores[camera_id] = scores.get(camera_id, 0) + score
    return scores

def score_simple(feature_tracks, track_points, poses):
    scores = score_using_count(feature_tracks, track_points, poses)
    max_score = -999
    best_camera_id = None

    for camera_id, score in scores.items():
        if score > max_score:
            max_score = score
            best_camera_id = camera_id
    return (best_camera_id, max_score)


def score_complicated(feature_tracks, track_points, poses, keypoints):
    image_width = 3024
    image_height = 4032

def scores(feature_tracks, track_points, poses):
    
    simple_scores = score_simple(feature_tracks, track_points, poses)
    print(f"Best camera ID: {simple_scores[0]}, Score: {simple_scores[1]}")

    
