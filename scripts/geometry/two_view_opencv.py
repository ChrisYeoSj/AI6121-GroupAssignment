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
import itertools
import functools
from dataclasses import dataclass, field

# Constants and Hyperparamters
OPENCV_METHOD = cv2.USAC_MAGSAC
MINIMUM_MATCHES = 30 #increased from 15, (colmap does 15)
MINIMUM_INLIERS = 15
 # we tested 1 to 5px, rotation change aginst 3px is under 0.03 degrees on every scene. 1 and 2px is worse
INLIER_THRESHOLD_PIXELS = 3.0

RANSAC_CONFIDENCE = 0.999
RANSAC_MAXIMUM_ITERATIONS = 1000
RANSAC_MINIMUM_ITERATIONS = 100
LOCAL_REFIT_THRESHOLD_MULTIPLES = (3.0, 2.0, 1.5, 1.0)

PANORAMIC_RATIO = 0.95
PLANAR_RATIO = 0.95

# smallest angle we'll accept, where depth error grows as the angle shrinks
MINIMUM_TRIANGULATION_ANGLE_DEGREES = 2.0

# seed rules handling
# 1 strict and a fallback for seeds
SEED_RULES = ("strict", "relaxed")
# at least 100 minimum good seed points to start a 3d map under the strict rule
MINIMUM_SEED_POINTS = 100

# minimum seed points of 50, for fallback jic.
FALLBACK_MINIMUM_SEED_POINTS = 50

# ratio with 1, is almost flat and ratio with 0 scene with plenty of depth
SEED_MAXIMUM_HOMOGRAPHY_RATIO = 0.75

# a seed should reach this median angle
# we tested 15 to 20 degrees and found 16 to be superior.
SEED_MINIMUM_ANGLE_DEGREES = 16.0

# data classes, just for clean housekeeping.
class NoSeedPairFound(Exception):
    """No pair is good enough to start a reconstruction from."""

@dataclass
class TwoViewResult:
    """Everything learned about one pair of images."""
    first_image: int
    second_image: int
    label: str  # general, planar, panoramic or rejected
    reason: str  # one plain sentence saying why it got that label
    number_of_matches: int
    essential_inliers: int = 0
    homography_inliers: int = 0
    rotation_inliers: int = 0
    rotation: np.ndarray = None # second camera relative to the first
    translation: np.ndarray = None # unit length
    points: np.ndarray = None # (P,3) good points, first camera's frame
    point_match_rows: np.ndarray = None # (P,) which row of the matches made each point
    median_triangulation_angle_degrees: float = 0.0
    pose_from: str = "" # essential, homography, or empty if there is no pose

    @property
    def number_of_points(self):
        return 0 if self.points is None else len(self.points)

    @property
    def reaches_seed_angle(self):
        """Is the median viewing angle at least SEED_MINIMUM_ANGLE_DEGREES?"""
        return self.median_triangulation_angle_degrees >= SEED_MINIMUM_ANGLE_DEGREES

    @property
    def seed_score(self):
        """number of good points, marked down if the viewing angle is below
        SEED_MINIMUM_ANGLE_DEGREES. If its at or above it, this is just the point count."""
        angle_quality = min(self.median_triangulation_angle_degrees / SEED_MINIMUM_ANGLE_DEGREES, 1.0)
        return self.number_of_points * angle_quality

@dataclass
class SeedChoice:
    """which pair won, and how"""
    seed: TwoViewResult
    rule: str                    # strict, relaxed, see SEED_RULES
    height_band: str             # for example high-low || unknown without height labels
    candidates_per_rule: dict    # {rule: number of pairs that pass it}
    best_per_band: dict          # {band: {"pair", "score", "candidates"}} under the winning rule
    note: str = ""               # jic


@dataclass
class InitialReconstruction:
    """the starting map from which the incremental stage grows from."""
    seed: TwoViewResult
    poses: dict                  # {image_id: (rotation, translation)}
    points: np.ndarray            # (P,3)
    observations: dict           # {image_id: (P,) keypoint index of each point}
    all_pair_results: dict = field(default_factory=dict)
    seed_choice: SeedChoice = None

# Housekeeping to keep results clean
def _pair_result(first_image, second_image, number_of_matches, label, reason, **values):
    return TwoViewResult(first_image, second_image, label, reason, number_of_matches, **values)

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
    """3d points from two cameras with cv2.triangulatePoints. Returns (N,3).
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

    it measures the same thing as essential_matrix_errors(), so inlier counts from
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

# Triangulation Code
def find_valid_points(camera_matrix, rotations, translations, normalised_per_camera, points,
                      threshold_pixels=INLIER_THRESHOLD_PIXELS,
                      minimum_angle_degrees=MINIMUM_TRIANGULATION_ANGLE_DEGREES):
    """we check which triangulated points can be trusted? Returns (N,) (bool)

    A point must be in front of every camera and land near where it was seen in
    every camera (L11 page 52), and be seen from a wide enough angle.
    """
    threshold = pixel_threshold_to_normalised_units(camera_matrix, threshold_pixels)
    points = np.asarray(points, dtype=np.float64)
    is_valid = np.all(np.isfinite(points), axis=1)
    finite_points = np.where(is_valid[:, None], points, 0.0)
    for rotation, translation, normalised_points in zip(rotations, translations, normalised_per_camera):
        in_camera = (np.asarray(rotation) @ finite_points.T).T + np.asarray(translation)
        depth = in_camera[:, 2]
        is_valid &= depth > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = in_camera[:, :2] / depth[:, None]
            errors = np.linalg.norm(projected - np.asarray(normalised_points)[:, :2], axis=1)
            is_valid &= errors < threshold
    angles = triangulation_angles_degrees(rotations, translations, finite_points)
    return is_valid & (angles >= minimum_angle_degrees)

def triangulation_angles_degrees(rotations, translations, points):
    """widesst angle at each point between any two cameras that see it, returns (N,)"""
    points = np.asarray(points, dtype=np.float64)
    camera_centres = [-np.asarray(rotation).T @ np.asarray(translation)
                      for rotation, translation in zip(rotations, translations)]
    widest_angles = np.zeros(len(points))
    for centre_a, centre_b in itertools.combinations(camera_centres, 2):
        with np.errstate(invalid="ignore"):
            direction_a = centre_a - points
            direction_b = centre_b - points
            # arctan2 stays exact for very small angles, where arccos does not.
            sine = np.linalg.norm(np.cross(direction_a, direction_b), axis=1)
            cosine = np.sum(direction_a * direction_b, axis=1)
            angles = np.degrees(np.arctan2(sine, cosine))
        angles[~np.isfinite(angles)] = 0.0
        widest_angles = np.maximum(widest_angles, angles)
    return widest_angles

def analyse_one_pair_of_image(camera_matrix, keypoints_first, keypoints_second, matches,
                 first_image=0, second_image=1, random_generator=None,
                 threshold_pixels=INLIER_THRESHOLD_PIXELS):

    if random_generator is None:
        random_generator = np.random.default_rng(0)

    number_of_matches = len(matches)

    result = functools.partial(_pair_result, first_image, second_image, number_of_matches)

    if number_of_matches < MINIMUM_MATCHES:
        return result("rejected", f"only {number_of_matches} matches, needs {MINIMUM_MATCHES}")

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
        return result("rejected", f"best model has only {best_model_inliers} inliers!", **counts)

    # check rotation first where a pure rotation also fits a homography perfectly
    if counts["rotation_inliers"] >= PANORAMIC_RATIO * best_model_inliers:
        return result("panoramic", "a pure rotation explains the matches", **counts)

    # check planar test, planar pair can still give correct pose and 3d points
    if counts["homography_inliers"] >= PLANAR_RATIO * max(counts["essential_inliers"], 1):
        # we will come back to this after everything is done
        return result("planar", "a homography explains the matches as well as E does", **counts)

    if essential_matrix is None:
        return result("rejected", "OpenCV found no essential matrix", **counts)

    # recover pose, it will try 4 poses from E adn keeps the one with the most inlier points in front of both cameras
    in_front, rotation, translation, _ = cv2.recoverPose(
        essential_matrix, pixels_first, pixels_second, camera_matrix,
        mask=essential_is_inlier.astype(np.uint8).reshape(-1, 1))

    # check in_front
    if in_front == 0:
        return result("rejected", "none of the 4 poses from the E matrix puts an inlier in front of both cameras",
                      **counts)

    # ravel and length must be 1, done as a precaution.
    translation = translation.ravel() / np.linalg.norm(translation)

    rotations = np.stack([np.eye(3), rotation])
    translations = np.stack([np.zeros(3), translation])
    inlier_rows = np.flatnonzero(essential_is_inlier)
    normalised_per_camera = np.stack([normalised_first_image_points[inlier_rows], normalised_second_image_points[inlier_rows]])
    points = triangulate_two_views(rotation, translation, normalised_per_camera[0], normalised_per_camera[1])
    is_valid = find_valid_points(camera_matrix, rotations, translations, normalised_per_camera,
                                 points, threshold_pixels)

    # judge parallex on points that passed every test except the angle.
    no_angle_limit = find_valid_points(camera_matrix, rotations, translations, normalised_per_camera,
                                       points, threshold_pixels, minimum_angle_degrees=0.0)

    # reject
    if not no_angle_limit.any():
        return result("rejected", "no match triangulates in front of both cameras", **counts)

    angles = triangulation_angles_degrees(rotations, translations, points[no_angle_limit])
    median_angle = float(np.median(angles))

    if median_angle < MINIMUM_TRIANGULATION_ANGLE_DEGREES:
        return result("panoramic", f"median triangulation angle is only {median_angle:.2f} deg",
                      median_triangulation_angle_degrees=median_angle, **counts)

    if is_valid.sum() < MINIMUM_INLIERS:
        return result("rejected", f"only {int(is_valid.sum())} good 3D points",
                      median_triangulation_angle_degrees=median_angle, **counts)

    return result("general", "the essential matrix explains the matches best",
                  rotation=rotation, translation=translation, pose_from="essential",
                  points=points[is_valid], point_match_rows=inlier_rows[is_valid],
                  median_triangulation_angle_degrees=median_angle, **counts)


def analyse_all_pairs_of_images(camera_matrix, keypoints_by_image, matches_by_pair, random_seed=0):
    """run analyse_one_pair_of_image on every pair, note the same random_seed gives the same answer"""
    pair_results = {}
    for pair in sorted(matches_by_pair):
        first_image, second_image = pair
        random_generator = np.random.default_rng([random_seed, first_image, second_image])
        pair_results[pair] = analyse_one_pair_of_image(camera_matrix, keypoints_by_image[first_image],
                                          keypoints_by_image[second_image], matches_by_pair[pair],
                                          first_image, second_image, random_generator)
    return pair_results

def passes_seed_rule(result, rule):
    """can this pair pass a seed rule a map under this rule? See SEED_RULES."""
    if rule == "strict":
        return (result.label == "general"
                and result.number_of_points >= MINIMUM_SEED_POINTS
                and result.homography_inliers < SEED_MAXIMUM_HOMOGRAPHY_RATIO * result.essential_inliers)
    if rule == "relaxed":
        return result.label == "general" and result.number_of_points >= FALLBACK_MINIMUM_SEED_POINTS
    raise ValueError(f"unknown seed rule {rule}")

def _seed_order(result):
    """seed score with the highest score wins. edge case handling of tie
      goes to the lowest image numbers, so this is replicable"""
    return (result.seed_score, -result.first_image, -result.second_image)

def height_band_of_pair(first_image, second_image, height_band_of_image=None):
    """just for record and experimentation. for example high-low | low-low, the two bands are sorted, so the order of images does not matter"""
    if not height_band_of_image:
        return "unknown"
    return "-".join(sorted((height_band_of_image[first_image], height_band_of_image[second_image])))

def choose_seed(pair_results, height_band_of_image=None):
    """pick the starting pair. try the strict rule, then the relaxed fallback."""
    candidates_per_rule = {rule: sum(passes_seed_rule(result, rule) for result in pair_results.values())
                           for rule in SEED_RULES}
    for rule in SEED_RULES:
        candidates = [result for result in pair_results.values() if passes_seed_rule(result, rule)]
        if not candidates:
            continue
        best_per_band = {}
        for result in sorted(candidates, key=_seed_order, reverse=True):
            band = height_band_of_pair(result.first_image, result.second_image, height_band_of_image)
            entry = best_per_band.setdefault(band, {"pair": [result.first_image, result.second_image],
                                                    "score": float(result.seed_score), "candidates": 0})
            entry["candidates"] += 1
        seed = max(candidates, key=_seed_order)
        return SeedChoice(seed, rule,
                          height_band_of_pair(seed.first_image, seed.second_image, height_band_of_image),
                          candidates_per_rule, best_per_band)
    raise NoSeedPairFound(
        f"no pair passes any seed rule. no general pair has {FALLBACK_MINIMUM_SEED_POINTS} good 3D points")

def initialise_reconstruction(camera_matrix, keypoints_by_image, matches_by_pair, random_seed=0,
                              height_band_of_image=None):
    """full initialisation where it label every pair, pick the seed, return the starting map"""
    pair_results = analyse_all_pairs_of_images(camera_matrix, keypoints_by_image, matches_by_pair, random_seed)
    seed_choice = choose_seed(pair_results, height_band_of_image)
    seed = seed_choice.seed
    seed_matches = matches_by_pair[(seed.first_image, seed.second_image)][seed.point_match_rows]
    return InitialReconstruction(
        seed=seed,
        poses={seed.first_image: (np.eye(3), np.zeros(3)),
               seed.second_image: (seed.rotation, seed.translation)},
        points=seed.points,
        observations={seed.first_image: seed_matches[:, 0],
                      seed.second_image: seed_matches[:, 1]},
        all_pair_results=pair_results,
        seed_choice=seed_choice)
