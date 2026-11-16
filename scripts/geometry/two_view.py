"""
Core Responsibilities of this two_view.py script is:
Initialisation and Triangulation for the Structure for Motion Pipeline.

Tasks:
Work out the position of 2-4 cameras from matched points alone, with no prior 3D knowledge
Turn matched points into actual 3D coordinates.


What this script needs:
camera_matrix   3x3 K matrix 

Lreferences
11. Epipolar & Stereo Vision - Prof Lu 2026
12. Multi-View Stereo -  Prof Lu 2026

L11 is refered to as L11 and L12 is refered to as L12 for brevity
"""
# imports
import numpy as np
import itertools

# CONSTANTS here
MINIMUM_MATCHES = 15 #same as colmap, should we increase?

 # we tested 1 to 5px, rotation change aginst 3px is under 0.03 degrees on every scene. 1 and 2px is worse
INLIER_THRESHOLD_PIXELS = 3.0

# RANSAC Constants
RANSAC_MAXIMUM_ITERATIONS = 1000
RANSAC_MINIMUM_ITERATIONS = 100
RANSAC_CONFIDENCE = 0.999

LOCAL_REFIT_THRESHOLD_MULTIPLES = (3.0, 2.0, 1.5, 1.0) # another hyperparameter to tune

POSE_POLISH_STEPS = 10

# Experiments Logs
"""
# Experiments Ideas we can test.
1. AB Test: Minimum Matches, can test a relaxed 10 | 15 | 30
2. with and without RANSAC
3. AB Tests: Pixel Threshold Hyperparameters
4. with MSAC and with plain RANSAC with inlier counting (keeping the model with the most inliers (L10 - Page 51)).
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
    """a sample can be degenerate, for example points on a line. skip those samples"""
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
    """Goal: Fit a model while ignoring bad matches. Refering to L10, pages 47-51"""
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

# Start of 8 Point Equation

def _normalising_transform(normalised_points):
    """Move points so their centre is at 0 and their mean squared distance is 2.

    This is the transform T on L10 page 42-44. Without it, the equations in the fit
    mix very different sizes of number, and small noise causes large errors
    Soruce: (L10 page 40-41).
    """
    centre = normalised_points[:, :2].mean(axis=0)
    mean_squared_distance = np.mean(np.sum((normalised_points[:, :2] - centre) ** 2, axis=1))
    scale = np.sqrt(2.0 / max(mean_squared_distance, 1e-24))
    return np.array([[scale, 0.0, -scale * centre[0]],
                     [0.0, scale, -scale * centre[1]],
                     [0.0, 0.0, 1.0]])

def decompose_essential_matrix(essential_matrix):
    """The four (rotation, translation) answers that give the same essential matrix.

    The lectures get the cameras from F up to a projective ambiguity instead (L11 p40-45). 
    We refer to Hartley and Zisserman result 9.19.
    """
    left_vectors, _, right_vectors_transposed = np.linalg.svd(essential_matrix)
    # Both must be true rotations, or the answers below would be mirror images.
    if np.linalg.det(left_vectors) < 0:
        left_vectors = -left_vectors
    if np.linalg.det(right_vectors_transposed) < 0:
        right_vectors_transposed = -right_vectors_transposed
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_a = left_vectors @ quarter_turn @ right_vectors_transposed
    rotation_b = left_vectors @ quarter_turn.T @ right_vectors_transposed
    translation = left_vectors[:, 2]
    return [(rotation_a, translation), (rotation_a, -translation),
            (rotation_b, translation), (rotation_b, -translation)]

def _essential_from_pose(rotation, translation):
    """E = [t]x R, where [t]x is the matrix form of the cross product with t.

    Lectures: L10 p25 (E = [T x] R), p26 (cross product as a matrix).
    """
    cross_product_matrix = np.array([[0.0, -translation[2], translation[1]],
                                     [translation[2], 0.0, -translation[0]],
                                     [-translation[1], translation[0], 0.0]])
    return cross_product_matrix @ rotation

def _polish_pose(rotation, translation, normalised_first, normalised_second):
    """Levenberg-Marquardt on the Sampson error of the inliers.

    An essential matrix has 5 degrees of freedom: 3 for rotation and 2 for the
    direction of translation. Each RANSAC sample uses only 8 noisy matches, so
    this step is what lets all of the inliers improve the answer.

    Lectures: L11 p53 names Levenberg-Marquardt, there for bundle adjustment. Here it is
    used on one pair's Sampson error.
    """
    def apply_step(step):
        new_rotation = _rotation_from_small_vector(step[:3]) @ rotation
        # Two directions at right angles to t. Moving along t changes nothing.
        _, _, right_vectors_transposed = np.linalg.svd(translation[None, :])
        new_translation = translation + step[3] * right_vectors_transposed[1] \
            + step[4] * right_vectors_transposed[2]
        return new_rotation, new_translation / np.linalg.norm(new_translation)

    def residuals_of(step):
        return _signed_essential_residuals(_essential_from_pose(*apply_step(step)),
                                           normalised_first, normalised_second)

    damping = 1e-3
    finite_difference = 1e-7
    for _ in range(POSE_POLISH_STEPS):
        residuals = residuals_of(np.zeros(5))
        cost = np.sum(residuals ** 2)
        jacobian = np.stack([(residuals_of(finite_difference * unit_step) - residuals)
                             / finite_difference for unit_step in np.eye(5)], axis=1)
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        improved = False
        while damping < 1e8:
            step = -np.linalg.solve(normal_matrix + damping * np.diag(np.diag(normal_matrix) + 1e-12),
                                    gradient)
            if np.sum(residuals_of(step) ** 2) < cost:
                rotation, translation = apply_step(step)
                damping = max(damping / 10.0, 1e-12)
                improved = True
                break
            damping *= 10.0
        if not improved:
            break
    return rotation, translation

def _signed_essential_residuals(essential_matrix, normalised_first, normalised_second):
    """Sampson distance with its sign kept. Gauss-Newton needs the sign."""
    line_in_second = (essential_matrix @ normalised_first.T).T
    line_in_first = (essential_matrix.T @ normalised_second.T).T
    algebraic_error = np.sum(normalised_second * line_in_second, axis=1)
    gradient_size = np.sqrt(line_in_second[:, 0] ** 2 + line_in_second[:, 1] ** 2
                            + line_in_first[:, 0] ** 2 + line_in_first[:, 1] ** 2)
    return algebraic_error / np.maximum(gradient_size, 1e-15)


def essential_matrix_errors(essential_matrix, normalised_first, normalised_second):
    """Sampson distance: a first-order estimate of how far a match is from fitting.

    Lectures: the lectures use the distance from a point to its epipolar line
    (L10 p40, p51b). The Sampson distance is not in the lectures. Source: Sampson 1982,
    Hartley and Zisserman ch. 11.4.
    """
    return np.abs(_signed_essential_residuals(essential_matrix, normalised_first, normalised_second))

def polish_essential_matrix(essential_matrix, normalised_first, normalised_second):
    """Improve E on the given matches, keeping it a true essential matrix."""
    # Any of the four answers gives the same E, so use the first.
    rotation, translation = decompose_essential_matrix(essential_matrix)[0]
    return _essential_from_pose(*_polish_pose(rotation, translation, normalised_first, normalised_second))


def essential_matrix_is_plausible(essential_matrix, normalised_first, normalised_second):
    """Do nearly all of this model's inliers land in front of both cameras?"""
    if len(normalised_first) < 5:
        return False
    _, _, in_front = choose_pose(essential_matrix, normalised_first, normalised_second)
    return in_front >= MINIMUM_FRACTION_IN_FRONT * len(normalised_first)

def fit_fundamental_matrix(pixels_first, pixels_second):
    """Normalised 8-point algorithm. Needs at least 8 matches.

    Works on any homogeneous points (x, y, 1). on pixels it gives F, and needs no K:
    for a correct match, pixel_second.T @ F @ pixel_first = 0. 
    On normalised image coordinates (K^-1 times pixels) the same fit gives E, as E = K.T @ F @ K
    (L10 page 52). fit_essential_sample() uses it that way.

    L10 page 33-37 (Wf = 0 solution found by SVD), page 38-39 (force rank 2),
    page 41-45 (normalised 8-point method). Also L11 page 37-39 and page 64.
    """
    transform_first = _normalising_transform(pixels_first)
    transform_second = _normalising_transform(pixels_second)
    conditioned_first = (transform_first @ pixels_first.T).T
    conditioned_second = (transform_second @ pixels_second.T).T
    equations = np.einsum("ni,nj->nij", conditioned_second, conditioned_first).reshape(-1, 9)
    _, _, right_vectors_transposed = np.linalg.svd(equations)
    matrix = right_vectors_transposed[-1].reshape(3, 3)
    # Every F has rank 2, because all its epipolar lines meet at one point. Noise
    # breaks this, so set the smallest singular value to 0.
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(matrix)
    matrix = left_vectors @ np.diag([singular_values[0], singular_values[1], 0.0]) @ right_vectors_transposed
    matrix = transform_second.T @ matrix @ transform_first
    return matrix / np.linalg.norm(matrix)

def fit_essential_sample(normalised_first, normalised_second):
    rough = fit_fundamental_matrix(normalised_first, normalised_second)
    return [polish_essential_matrix(rough, normalised_first, normalised_second)]

def finish_essential_matrix(essential_matrix, is_inlier, normalised_first, normalised_second, threshold):
    """Make RANSAC's winner a true E, then check it. Returns (E, is_inlier), or (None, all False).

    Lectures: L10 page 51 step (4) re-estimates on all inliers at the end, and page 52 says
    E is found with the same method. The polish itself is _polish_pose().
    """
    if essential_matrix is None:
        return essential_matrix, is_inlier
    for _ in range(2):
        essential_matrix = polish_essential_matrix(essential_matrix, normalised_first[is_inlier],
                                                   normalised_second[is_inlier])
        is_inlier = essential_matrix_errors(essential_matrix, normalised_first, normalised_second) < threshold
    if not essential_matrix_is_plausible(essential_matrix, normalised_first[is_inlier],
                                         normalised_second[is_inlier]):
        return None, np.zeros(len(normalised_first), dtype=bool)
    return essential_matrix, is_inlier


def analyse_one_pair_of_image(camera_matrix, matches, keypoints_first, keypoints_second,
                              random_generator,
                              threshold_pixels=INLIER_THRESHOLD_PIXELS):

    # seed handling for reproduction
    if random_generator is None:
        random_generator = np.random.default_rng(0)

    number_of_matches = len(matches)

    if number_of_matches < MINIMUM_MATCHES:
        raise ValueError(f"Less than Minimum Matches of ${MINIMUM_MATCHES}")

    # COLMAP matches intially, while we wait for asmita.
    normalised_first_image_points = calculate_normalised_image_coordinates(camera_matrix, np.asarray(keypoints_first)[matches[:, 0]])
    normalised_second_image_points = calculate_normalised_image_coordinates(camera_matrix, np.asarray(keypoints_second)[matches[:, 1]])
    threshold = pixel_threshold_to_normalised_units(camera_matrix, threshold_pixels)

    # essential matrix E
    essential_matrix, essential_is_inlier = ransac(
    fit_essential_sample, polish_essential_matrix, essential_matrix_errors, 8,
        normalised_first_image_points, normalised_second_image_points, threshold, random_generator,
        is_plausible=essential_matrix_is_plausible)

    essential_matrix, essential_is_inlier = finish_essential_matrix(
        essential_matrix, essential_is_inlier, normalised_first_image_points, normalised_second_image_points, threshold)

    # homography
    homography, homography_is_inlier = ransac(
        lambda first, second: [fit_homography(first, second)],
        lambda _model, first, second: fit_homography(first, second),
        homography_errors, 4, normalised_first, normalised_second, threshold, random_generator)
