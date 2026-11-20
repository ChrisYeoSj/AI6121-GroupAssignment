import numpy as np

"""
Reference: 
Schonberger and Frahm, Structure-from-Motion Revisited, 2016
https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf

Follows Section 4.2 as reference
Quoted from the paper:
A popular strategy is to choose the image that sees most
triangulated points with the aim of minimizing the un-
certainty in camera resection. Haner et al. [24] propose an
uncertainty-driven approach that minimizes the reconstruc-
tion error. Usually, the camera that sees the largest number
of triangulated points is chosen...
"""

def score_using_count(feature_tracks, track_points, poses):
    scores = {} 
    def score_for_single_3d_point(camera_id):
        if camera_id in poses:
            return 0
        else:
            return 1

    for track_id, points_in_track in feature_tracks.items():
        track_points_length = len(track_points.get(track_id, []))
        if track_points_length == 1:
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

    for tuple_item in scores.items():
        camera_id = tuple_item[0]
        score = tuple_item[1]
        if score > max_score:
            max_score = score
            best_camera_id = camera_id
    return (best_camera_id, max_score)

def grid(feature_tracks, track_points, poses, keypoints):
    
    def score_for_single_3d_point(camera_id):
        if camera_id in poses:
            return 0
        else:
            return 1
    def get_x_y_from_keypoint(camera_id, keypoint_id):
        x = keypoints[camera_id][keypoint_id][0]
        y = keypoints[camera_id][keypoint_id][1]
        return (x, y)
    
    scores = {}
    for track_id, points_in_track in feature_tracks.items():
        track_points_length = len(track_points.get(track_id, []))
        if track_points_length == 1:
            for tuple_item in points_in_track:
                camera_id = tuple_item[0]

                score = score_for_single_3d_point(camera_id)
                if score == 1:
                    get_coordinates = get_x_y_from_keypoint(camera_id, int(tuple_item[1]))

                    if camera_id not in scores:
                        scores[camera_id] = []
                    scores[camera_id].append(get_coordinates)
    return scores


def calc_score_grid(camera_points, width, height):
    def get_cell_size(size, width, height):
        cell_width = width / size
        cell_height = height / size
        
        return (cell_width, cell_height)
    def get_cells(x, y, width_cell, height_cell, grid_size):
        cell_x = int(x / width_cell)
        if grid_size - 1 <= cell_x:
            cell_x = grid_size - 1
        if cell_x < 0:
            cell_x = 0
        
        cell_y = int(y / height_cell)
        if grid_size - 1 <= cell_y:
            cell_y = grid_size - 1
        if cell_y < 0:
            cell_y = 0
        return (cell_x, cell_y)

    cells = {}
    papers_num_lvls = 6
    
    grid_size = 2 ** papers_num_lvls

    cell_size = get_cell_size(grid_size, width, height)

    width_cell = cell_size[0]
    height_cell = cell_size[1]

    for tuple_item in camera_points.items():
        camera_id = tuple_item[0]
        points = tuple_item[1]
        for point in points:
            cell_x_y = get_cells(point[0], point[1], width_cell, height_cell, grid_size)

            cell_x = cell_x_y[0]
            cell_y = cell_x_y[1]
            # Colmap implementation in C++ VisibilityPyramid::VisibilityPyramid
            # Line 13 in reference file
            for i in range(papers_num_lvls - 1, -1, -1):
                if camera_id in cells:
                    if i in cells[camera_id]:
                        cells[camera_id][i].add((cell_x, cell_y))
                    else:
                        cells[camera_id][i] = set()
                        cells[camera_id][i].add((cell_x, cell_y))
                else:
                    cells[camera_id] = {}
                    cells[camera_id][i] = set()
                    cells[camera_id][i].add((cell_x, cell_y))
                cell_x = cell_x // 2
                cell_y = cell_y // 2
    return cells

def calc_final_scores(cells):
    def calc_per_level(level, cells, grid_size):
        num_cells = len(cells)
        level_score = num_cells * (grid_size ** 2)
        return level_score

    final_scores = {}
    for tuple_item in cells.items():
        camera_id = tuple_item[0]
        levels = tuple_item[1]
        total_score = 0
        for level, occup in levels.items():
            level_score = calc_per_level(level, occup, 2 ** (level + 1))
            total_score += level_score
        final_scores[camera_id] = total_score
    return final_scores

"""
References:
https://computervisiononline.com/explains/colmap/
Quote from article / website:
Next-best-view selection. Each candidate image is scored by how many already triangulated points it sees and how evenly they cover it. 
The paper counts occupied cells in a pyramid of image grids with Kl = 2^l cells per side at level l, weighted by K^2l,
so that spread-out poins score higher than clustered ones.

"""
def score_complicated(feature_tracks, track_points, poses, keypoints):
    
    camera_points = grid(feature_tracks, track_points, poses, keypoints)
    scores = calc_score_grid(camera_points, 3024, 4032)
    final_scores = calc_final_scores(scores)
    return final_scores

def scores(feature_tracks, track_points, poses, keypoints):
    
    simple_scores = score_simple(feature_tracks, track_points, poses)
    complicated_scores = score_complicated(feature_tracks, track_points, poses, keypoints)

    return (simple_scores, complicated_scores)

    
