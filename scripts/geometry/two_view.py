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

INLIER_THRESHOLD_PIXELS = 3.0 # hyperparameter - to tune

# RANSAC Constants
RANSAC_MAXIMUM_ITERATIONS = 1000
RANSAC_MINIMUM_ITERATIONS = 100
RANSAC_CONFIDENCE = 0.999

LOCAL_REFIT_THRESHOLD_MULTIPLES = (3.0, 2.0, 1.5, 1.0) # another hyperparameter to tune


# Experiments Logs
"""
# Experiments Ideas we can test.
1. AB Test: Minimum Matches, can test a relaxed 10 | 15 | 30
2. with and without RANSAC
3. AB Tests: Pixel Threshold Hyperparameters
4. with MSAC and with plain RANSAC with inlier counting (keeping the model with the most inliers (Lecture 10 - Page 51)).
5. Single Refit or using LO-RANSAC's approach.

"""

# following the slide's Normalize 𝒑 and 𝒑’ (projection of 𝑃) with the camera's parameters removed.
# normalise pixels with K: p_c = K^-1 p. Returns (N,3), with the last value equal to 
# slides call p_c the normalised image coordinates (L10 p52)
# 
def calculate_normalised_image_coordinates(camera_matrix, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous_pixels = np.column_stack([pixels, np.ones(len(pixels))])
    return (np.linalg.inv(camera_matrix) @ homogeneous_pixels.T).T

def pixel_threshold_to_normalised_units(camera_matrix, threshold_pixels):
    """one pixel is about 1/f in normalised image coordinates, so thresholds are divided by f"""
    # 0,0 => is fx, and 1,1 => is fy, we do this because fx = 3079.19 and fy = 3055.22
    # averaging here as it doesnt have to be precise, as its a cutoff
    average_focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0
    return threshold_pixels / average_focal_length


# RANSAC Functions
def _iterations_for_confidence(inlier_fraction, sample_size):
    """how many tries it takes to probably draw one sample with no bad matches.
    Lectures Source: L10 p51 says to repeat for N rounds but not how big N is. 
    source: Fischler and Bolles 1981
    """
    all_good_probability = inlier_fraction ** sample_size
    if all_good_probability >= 1.0:
        return 1
    if all_good_probability <= 0.0:
        return RANSAC_MAXIMUM_ITERATIONS
    # log1p, because 1 - p rounds to exactly 1 when p is tiny, and log(1) is 0.
    needed = np.log1p(-RANSAC_CONFIDENCE) / np.log1p(-all_good_probability)
    return int(min(np.ceil(needed), RANSAC_MAXIMUM_ITERATIONS))

def _try_to_solve(solve_sample, normalised_first, normalised_second):
    """A sample can be degenerate, for example points on a line. Skip those."""
    try:
        return [model for model in solve_sample(normalised_first, normalised_second)
                if np.all(np.isfinite(model))]
    except np.linalg.LinAlgError:
        return []

def _model_cost(errors, threshold):
    """MSAC score, lower is better. An inlier costs its squared error, an outlier a flat amount.

    Counting inliers alone cannot separate a poor model from an exact one if
    both fit every match within the threshold. MSAC helps. 

    The lecture shows us to keep the model with the most inliers (L10 page51).
    We experimented with both recommendation from Slides and MSAC (Torr and Zisserman 2000).
    """
    return float(np.sum(np.minimum(errors, threshold) ** 2))

def _refit_on_inliers(refit_model, model_errors, sample_size, normalised_first, normalised_second,
                      threshold, model, cost):
    """Local optimisation: refit, tightening the threshold each round and then keep the best result.

    Lectures: L10 p51 step (iv) on a single refit, on all inliers, at the end.
    There is also an approach to refit at every new best model. Source: Chum et al. 2003 (LO-RANSAC).
    We test both in experiment.
    """
    best_model, best_cost = model, cost
    for threshold_multiple in LOCAL_REFIT_THRESHOLD_MULTIPLES:
        is_near = model_errors(model, normalised_first, normalised_second) < threshold * threshold_multiple
        if is_near.sum() <= sample_size:
            break
        try:
            refit = refit_model(model, normalised_first[is_near], normalised_second[is_near])
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(refit)):
            break
        model = refit
        refit_cost = _model_cost(model_errors(model, normalised_first, normalised_second), threshold)
        if refit_cost < best_cost:
            best_model, best_cost = model, refit_cost
    best_is_inlier = model_errors(best_model, normalised_first, normalised_second) < threshold
    return best_model, best_cost, best_is_inlier

def ransac(solve_sample, refit_model, model_errors, sample_size, normalised_first, normalised_second,
           threshold, random_generator, is_plausible=None):
    """Goal: Fit a model while ignoring bad matches. Refering to Lecture 10, pages 47-51"""
    number_of_matches = len(normalised_first)
    best_model = None
    best_cost = np.inf
    # True for each match that fits best_model, False otherwise. One value per match.
    best_is_inlier = np.zeros(number_of_matches, dtype=bool)
    if number_of_matches < sample_size:
        return best_model, best_is_inlier

    iterations_needed = RANSAC_MAXIMUM_ITERATIONS
    iteration = 0
    while iteration < iterations_needed:
        iteration += 1
        sample = random_generator.choice(number_of_matches, sample_size, replace=False)
        for model in _try_to_solve(solve_sample, normalised_first[sample], normalised_second[sample]):
            errors = model_errors(model, normalised_first, normalised_second)
            cost = _model_cost(errors, threshold)
            if cost < best_cost:
                refit, refit_cost, refit_is_inlier = _refit_on_inliers(
                    refit_model, model_errors, sample_size, normalised_first, normalised_second,
                    threshold, model, cost)
                if is_plausible is not None and not is_plausible(
                        refit, normalised_first[refit_is_inlier], normalised_second[refit_is_inlier]):
                    continue
                best_model, best_cost, best_is_inlier = refit, refit_cost, refit_is_inlier
                iterations_needed = max(
                    RANSAC_MINIMUM_ITERATIONS,
                    _iterations_for_confidence(best_is_inlier.mean(), sample_size))
    return best_model, best_is_inlier

# END OF RANSAC Functions

def analyse_one_pair_of_image(camera_matrix, matches, keypoints_first, keypoints_second,
                              random_generator,
                              threshold_pixels=INLIER_THRESHOLD_PIXELS):

    # seed handling for reproduction
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    number_of_matches = len(matches)

    if number_of_matches < MINIMUM_MATCHES:
        raise ValueError(f"Less than Minimum Matches of ${MINIMUM_MATCHES}")

    first_image_points = calculate_normalised_image_coordinates(camera_matrix, keypoints_first)
    second_image_points = calculate_normalised_image_coordinates(camera_matrix, keypoints_second)



