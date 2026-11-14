"""
Core Responsibilities of this two_view.py script is:
Initialisation and Triangulation for the Structure for Motion Pipeline.

Tasks:
Work out the position of 2-4 cameras from matched points alone, with no prior 3D knowledge
Turn matched points into actual 3D coordinates.


What this script needs:
camera_matrix   3x3 K matrix 

Lecture references
11. Epipolar & Stereo Vision - Prof Lu 2026
12. Multi-View Stereo -  Prof Lu 2026
"""
# imports
import numpy as np


# CONSTANTS here
MINIMUM_MATCHES = 15 #same as colmap, should we increase?

# Experiments Logs
"""
# Experiments Ideas we can test.
1. AB Test: Minimum Matches, can test a relaxed 10 | 15 | 30
2. with and without RANSAC
3. 

"""

# following the slide's Normalize 𝒑 and 𝒑’ (projection of 𝑃) with the camera's parameters removed.
# normalise pixels with K: p_c = K^-1 p. Returns (N,3), with the last value equal to 
# slides call p_c the normalised image coordinates (L10 p52)
# 
def calculate_normalised_image_coordinates(camera_matrix, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous_pixels = np.column_stack([pixels, np.ones(len(pixels))])
    return (np.linalg.inv(camera_matrix) @ homogeneous_pixels.T).T




def analyse_one_pair_of_image(camera_matrix, matches, random_generator, keypoints_first, keypoints_second):

    # seed handling for reproduction
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    number_of_matches = len(matches)

    if number_of_matches < MINIMUM_MATCHES:
        raise ValueError(f"Less than Minimum Matches of ${MINIMUM_MATCHES}")

    first_image_points = calculate_normalised_image_coordinates(camera_matrix, keypoints_first)
    second_image_points = calculate_normalised_image_coordinates(camera_matrix, keypoints_second)



