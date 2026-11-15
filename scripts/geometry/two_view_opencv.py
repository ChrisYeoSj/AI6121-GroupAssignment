"""
Core Responsibilities of this two_view.py script is:
Initialisation and Triangulation for the Structure for Motion Pipeline.

Tasks:
Work out the position of 2-4 cameras from matched points alone, with no prior 3D knowledge
Turn matched points into actual 3D coordinates.

What this script needs:
camera_matrix   3x3 K matrix 

Lecture References
11. Epipolar & Stereo Vision - Prof Lu 2026
12. Multi-View Stereo -  Prof Lu 2026

L11 is refered to as L11 and L12 is refered to as L12 for brevity.

Checked with Prof Lu regarding the cv2 usage and he ok'd it.
"""

import cv2
import numpy as np

# Constants and Hyperparamters
OPENCV_METHOD = cv2.USAC_MAGSAC
MINIMUM_MATCHES = 15 #same as colmap, should we increase?
MINIMUM_INLIERS = 15
 # we tested 1 to 5px, rotation change aginst 3px is under 0.03 degrees on every scene. 1 and 2px is worse
INLIER_THRESHOLD_PIXELS = 3.0

RANSAC_CONFIDENCE = 0.999
RANSAC_MAXIMUM_ITERATIONS = 1000
RANSAC_MINIMUM_ITERATIONS = 100
LOCAL_REFIT_THRESHOLD_MULTIPLES = (3.0, 2.0, 1.5, 1.0)

PANORAMIC_RATIO = 0.95
PLANAR_RATIO = 0.95

# Seed Handling
def _seed_opencv(random_generation):
    """For reproducibility"""
    cv2.setRNGSeed(int(random_generation.integers(0, 2 ** 31-1)))

def _random_order(random_generation, number_of_matches):
    """A random order for the matches, as OpenCV seems to ignores cv2.setRNGSeed"""
    return random_generation.permutation(number_of_matches)

def _pixels(keypoints, rows):
    return np.ascontiguousarray(np.asarray(keypoints, dtype=np.float64)[rows])

def _first_matrix(matrix):
    """Edgecase handling, where OpenCV can return several 3x3 answers stacked as a (3k, 3). Keep the first, or None."""
    if matrix is None or matrix.shape[0] < 3:
        return None
    return matrix[:3]

def _model_cost(errors, threshold):
    """MSAC score, lower is better
    The lecture shows us to keep the model with the most inliers (L10 page 51).
    We experimented with both recommendation from Slides and MSAC (Torr and Zisserman 2000).
    """
    return float(np.sum(np.minimum(errors, threshold) ** 2))

def _iterations_for_confidence(inlier_fraction, sample_size):
    """how many tries it takes to probably draw one sample with no bad matches (Fischler and Bolles 1981)."""
    all_good_probability = inlier_fraction ** sample_size
    if all_good_probability >= 1.0:
        return 1
    if all_good_probability <= 0.0:
        return RANSAC_MAXIMUM_ITERATIONS
    needed = np.log1p(-RANSAC_CONFIDENCE) / np.log1p(-all_good_probability)
    return int(min(np.ceil(needed), RANSAC_MAXIMUM_ITERATIONS))

def triangulate_two_views(rotation, translation, normalised_first, normalised_second):
    """3D points from two cameras with cv2.triangulatePoints. Returns (N,3).

    The first camera is [I | 0] and the second is [R | t]. Both work on normalised
    image coordinates, so K is not needed. A point with a last coordinate of 0
    comes back as inf.

    When the lines of sight are parallel, OpenCV still returns a finite point,
    with no meaning. Its triangulation angle is 0, so find_valid_points() rejects it.
    """
    first_camera = np.hstack([np.eye(3), np.zeros((3, 1))])
    second_camera = np.hstack([np.asarray(rotation, dtype=np.float64),
                               np.asarray(translation, dtype=np.float64).reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(first_camera, second_camera,
                                        np.ascontiguousarray(np.asarray(normalised_first)[:, :2].T),
                                        np.ascontiguousarray(np.asarray(normalised_second)[:, :2].T))
    last = homogeneous[3]
    points = np.full((homogeneous.shape[1], 3), np.inf)
    usable = np.abs(last) > 1e-12 * np.abs(homogeneous[:3]).max(axis=0).clip(min=1e-300)
    points[usable] = (homogeneous[:3, usable] / last[usable]).T
    return points

# following the slide's Normalize 𝒑 and 𝒑’ (projection of 𝑃) with the camera's parameters removed.
# normalise pixels with K: p_c = K^-1 p. Returns (N,3), with the last value equal to 
# slides call p_c the normalised image coordinates (L10 p52)
def calulate_pixels_to_normalised_image_coordinates(camera_matrix, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous_pixels = np.column_stack([pixels, np.ones(len(pixels))])
    return (np.linalg.inv(camera_matrix) @ homogeneous_pixels.T).T

def pixel_threshold_to_normalised_units(camera_matrix, threshold_pixels):
    """one pixel is about 1/f in normalised image coordinates, so thresholds are divided by f"""
    # 0,0 => is fx, and 1,1 => is fy, we do this, in our case, because fx = 3079.19 and fy = 3055.22
    # averaging here as it doesnt have to be precise, as its a cutoff
    average_focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0
    return threshold_pixels / average_focal_length

# wrapper function for Essential Matrix
def _fit_essential_matrix(camera_matrix, pixels_first, pixels_second, threshold_pixels, random_generator):
    """E with cv2.findEssentialMat (5-point solver inside OPENCV_METHOD). Returns E or None.
    """

    order = _random_order(random_generator, len(pixels_first))
    essential_matrix, _ = cv2.findEssentialMat(pixels_first[order], pixels_second[order], camera_matrix,
                                               method=OPENCV_METHOD, prob=RANSAC_CONFIDENCE,
                                               threshold=threshold_pixels,
                                               maxIters=RANSAC_MAXIMUM_ITERATIONS)
    return _first_matrix(essential_matrix)

# wrapper Function for Homography H
def _fit_homography(camera_matrix, pixels_first, pixels_second, threshold_pixels, random_generator):
    """Wrapper function for H with cv2.findHomography, moved into normalised image coordinates. Returns H or None.
    """
    order = _random_order(random_generator, len(pixels_first))
    homography, _ = cv2.findHomography(pixels_first[order], pixels_second[order], OPENCV_METHOD,
                                       threshold_pixels, maxIters=RANSAC_MAXIMUM_ITERATIONS,
                                       confidence=RANSAC_CONFIDENCE)
    if homography is None:
        return None
    return np.linalg.inv(camera_matrix) @ homography @ camera_matrix

def homography_errors(matrix, normalised_first, normalised_second):
    """sampson distance for a homography. This also works for a pure rotation.

    It measures the same thing as essential_matrix_errors(), so inlier counts from
    the two can be compared fairly.
    """
    x_second, y_second = normalised_second[:, 0], normalised_second[:, 1]
    mapped = (matrix @ normalised_first.T).T
    mapped_x, mapped_y, mapped_w = mapped[:, 0], mapped[:, 1], mapped[:, 2]

    # the two equations a perfect match satisfies, each equal to 0.
    error_a = mapped_y - y_second * mapped_w
    error_b = x_second * mapped_w - mapped_x

    # how each equation changes as each of the 4 coordinates moves.
    gradient_a = np.stack([matrix[1, 0] - y_second * matrix[2, 0],
                           matrix[1, 1] - y_second * matrix[2, 1],
                           np.zeros(len(mapped_w)), -mapped_w], axis=1)
    gradient_b = np.stack([x_second * matrix[2, 0] - matrix[0, 0],
                           x_second * matrix[2, 1] - matrix[0, 1],
                           mapped_w, np.zeros(len(mapped_w))], axis=1)

    # written out: errors.T @ inverse(gradients @ gradients.T) @ errors.
    a_a = np.sum(gradient_a * gradient_a, axis=1)
    a_b = np.sum(gradient_a * gradient_b, axis=1)
    b_b = np.sum(gradient_b * gradient_b, axis=1)
    determinant = a_a * b_b - a_b * a_b
    with np.errstate(divide="ignore", invalid="ignore"):
        squared_distance = (b_b * error_a ** 2 - 2 * a_b * error_a * error_b
                            + a_a * error_b ** 2) / determinant
        errors = np.sqrt(squared_distance)
    errors[~np.isfinite(errors)] = np.inf
    return errors

def essential_matrix_errors(essential_matrix, normalised_first, normalised_second):
    """sampson distance: a first-order estimate of how far a match is from fitting E.
    taken from: Sampson 1982 and Hartley and Zisserman chapter 11.4.
    """
    line_in_second = (essential_matrix @ normalised_first.T).T
    line_in_first = (essential_matrix.T @ normalised_second.T).T
    algebraic_error = np.sum(normalised_second * line_in_second, axis=1)
    gradient_size = np.sqrt(line_in_second[:, 0] ** 2 + line_in_second[:, 1] ** 2
                            + line_in_first[:, 0] ** 2 + line_in_first[:, 1] ** 2)
    return np.abs(algebraic_error / np.maximum(gradient_size, 1e-15))


def fit_rotation(normalised_first, normalised_second):
    """best pure rotation from the first image's points to the second's (Kabsch 1976). Needs 2 matches."""
    unit_first = normalised_first / np.linalg.norm(normalised_first, axis=1, keepdims=True)
    unit_second = normalised_second / np.linalg.norm(normalised_second, axis=1, keepdims=True)
    left_vectors, _, right_vectors_transposed = np.linalg.svd(unit_second.T @ unit_first)
    # block mirror images. a reflection has determinant -1 and is not a rotation.
    reflection_fix = np.diag([1.0, 1.0, np.linalg.det(left_vectors @ right_vectors_transposed)])
    return left_vectors @ reflection_fix @ right_vectors_transposed

def _refit_rotation(rotation, normalised_first, normalised_second, threshold, cost):
    """Refit on the inliers, tightening the threshold each round. Keep the best (Chum et al. 2003)."""
    best_rotation, best_cost = rotation, cost
    for threshold_multiple in LOCAL_REFIT_THRESHOLD_MULTIPLES:
        is_near = homography_errors(rotation, normalised_first, normalised_second) < threshold * threshold_multiple
        if is_near.sum() <= 2:
            break
        try:
            refit = fit_rotation(normalised_first[is_near], normalised_second[is_near])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(refit)):
            break
        rotation = refit
        refit_cost = _model_cost(homography_errors(rotation, normalised_first, normalised_second), threshold)
        if refit_cost < best_cost:
            best_rotation, best_cost = rotation, refit_cost
    return best_rotation, best_cost, homography_errors(best_rotation, normalised_first, normalised_second) < threshold

def _rotation_ransac(normalised_first, normalised_second, threshold, random_generator):
    """RANSAC for a pure rotation, where we need to catch pairs where the camera turned but didnt move"""
    number_of_matches = len(normalised_first)
    best_cost = np.inf
    best_is_inlier = np.zeros(number_of_matches, dtype=bool)
    if number_of_matches < 2:
        return best_is_inlier
    iterations_needed = RANSAC_MAXIMUM_ITERATIONS
    iteration = 0
    while iteration < iterations_needed:
        iteration += 1
        sample = random_generator.choice(number_of_matches, 2, replace=False)
        try:
            rotation = fit_rotation(normalised_first[sample], normalised_second[sample])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(rotation)):
            continue
        cost = _model_cost(homography_errors(rotation, normalised_first, normalised_second), threshold)
        if cost < best_cost:
            _, best_cost, best_is_inlier = _refit_rotation(rotation, normalised_first, normalised_second,
                                                           threshold, cost)
            iterations_needed = max(RANSAC_MINIMUM_ITERATIONS,
                                    _iterations_for_confidence(best_is_inlier.mean(), 2))
    return best_is_inlier

def analyse_one_pair_of_image(camera_matrix, keypoints_first, keypoints_second, matches,
                 first_image=0, second_image=1, random_generator=np.random.default_rng(0),
                 threshold_pixels=INLIER_THRESHOLD_PIXELS):

    number_of_matches = len(matches)

    if number_of_matches < MINIMUM_MATCHES:
        raise ValueError(f"Less than Minimum Matches of ${MINIMUM_MATCHES}")

    pixels_first = _pixels(keypoints_first, matches[:, 0])
    pixels_second = _pixels(keypoints_second, matches[:, 1])

    # COLMAP matches intially, while we wait for asmita.
    normalised_first_image_points = calulate_pixels_to_normalised_image_coordinates(camera_matrix, pixels_first)
    normalised_second_image_points = calulate_pixels_to_normalised_image_coordinates(camera_matrix, pixels_second)
    threshold = pixel_threshold_to_normalised_units(camera_matrix, threshold_pixels)

    no_inliers_fallback = np.zeros(number_of_matches, dtype=bool)

    essential_matrix = _fit_essential_matrix(camera_matrix, pixels_first, pixels_second, 
                                             threshold_pixels, random_generator)

    homography = _fit_homography(camera_matrix, pixels_first, pixels_second, threshold_pixels,
                                 random_generator)
    rotation_is_inlier = _rotation_ransac(normalised_first_image_points, normalised_second_image_points, threshold, random_generator)

    essential_is_inlier = (no_inliers_fallback if essential_matrix is None else
                           essential_matrix_errors(essential_matrix, normalised_first_image_points, normalised_second_image_points)
                           < threshold)

    homography_is_inlier = (no_inliers_fallback if homography is None else
                            homography_errors(homography, normalised_first_image_points, normalised_second_image_points) < threshold)

    counts = dict(essential_inliers=int(essential_is_inlier.sum()),
                  homography_inliers=int(homography_is_inlier.sum()),
                  rotation_inliers=int(rotation_is_inlier.sum()))

    best_model_inliers = max(counts["essential_inliers"], counts["homography_inliers"])

    # checks
    if best_model_inliers < MINIMUM_INLIERS:
        raise ValueError(f"This best model has only {best_model_inliers} inliers!")

    # check rotation first where a pure rotation also fits a homography perfectly
    if counts["rotation_inliers"] >= PANORAMIC_RATIO * best_model_inliers:
        return ValueError(f"note: a pure rotation explains the matches")

    # check planar test, planar pair can still give correct pose and 3d points
    if counts["homography_inliers"] >= PLANAR_RATIO * max(counts["essential_inliers"], 1):
        # we will come back to this after everything is done
        return ValueError(f"note: a homography explains the matches as well as E does.")

