"""
integration with calibration (john's code), features (asmita's code) and incremental reconstruction (emile's code)

how to run:
`python scripts/geometry/initialise_from_features.py [scenes] [--calibration PATH] [--random-seed N] [--incremental]`

it will run on all 4 rock scenes, if no scenes are given,

what it does for each scene:
1. loads asmita's keypoints, matches and tracks with load.py
(scripts/correspondence_search/load.py), and john's K and distortion
(results/calibration/calibration.json).
2. removes the lens distortion from every keypoint. asmita's keypoints are raw
pixels, but two_view_opencv.py and emile's code expect undistorted ones.
3. runs initialise_reconstruction() to build the starting map.
4. with --incremental, runs emile's entrypoint() to add the other cameras.

asmita's keypoints and K are both for the upright photo, so no turning is needed
"""

import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "correspondence_search"))

from load import load
from two_view_opencv import initialise_reconstruction

SCENES = ["yunnan_rock_1", "yunnan_rock_1_lower_height", "yunnan_rock_2", "yunnan_rock_2_lower_height"]
FEATURES_FOLDER = os.path.join(_ROOT, "results", "correspondence_search")
INCREMENTAL_FOLDER = os.path.join(_ROOT, "code", "icr")
CALIBRATION_PATH = os.path.join(_ROOT, "results", "calibration", "calibration.json")

# the default 5 iterations of cv2.undistortPoints are not enough near the lens corners
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)


def undistort_keypoints(camera_matrix, distortion, keypoints):
    """raw pixels (N,2) -> undistorted pixels (N,2), with the same K."""
    if len(keypoints) == 0:
        return np.zeros((0, 2))
    undistorted = cv2.undistortPoints(np.asarray(keypoints, dtype=np.float64).reshape(-1, 1, 2),
                                      camera_matrix, distortion, P=camera_matrix,
                                      criteria=UNDISTORT_CRITERIA)
    return undistorted.reshape(-1, 2)


def _load_undistorted(scene, calibration_path, features_folder):
    """asmita's load() output and undistorted keypoints"""
    if not os.path.isfile(calibration_path):
        raise FileNotFoundError(f"no calibration at {calibration_path}. Pass --calibration.")
    loaded = load(scene, out=features_folder, calib=calibration_path)
    keypoints_by_image = {image: undistort_keypoints(loaded["K"], loaded["dist"], keypoints)
                          for image, keypoints in enumerate(loaded["kpts"])}
    return loaded, keypoints_by_image


def load_features(scene, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """returns (camera_matrix, keypoints_by_image, matches_by_pair, image_names)"""
    loaded, keypoints_by_image = _load_undistorted(scene, calibration_path, features_folder)
    matches_by_pair = {pair: np.asarray(matches, dtype=np.int64) for pair, matches in loaded["matches"].items()}
    return loaded["K"], keypoints_by_image, matches_by_pair, loaded["image_names"]


def load_for_incremental(scene, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """one scene as the dict emile's entrypoint() reads, with undistorted keypoints"""
    loaded, keypoints_by_image = _load_undistorted(scene, calibration_path, features_folder)
    return dict(K=loaded["K"],
                kpts=[keypoints_by_image[image] for image in range(len(loaded["image_names"]))],
                matches={pair: np.asarray(matches, dtype=np.int64) for pair, matches in loaded["matches"].items()},
                tracks=dict(enumerate(loaded["tracks"])),
                image_names=loaded["image_names"])


def initialise_scene(scene, random_seed=0, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """load one scene and build its starting map. returns (InitialReconstruction, image_names)."""
    camera_matrix, keypoints_by_image, matches_by_pair, image_names = load_features(
        scene, calibration_path, features_folder)
    start = initialise_reconstruction(camera_matrix, keypoints_by_image, matches_by_pair, random_seed)
    return start, image_names


def run_incremental(scene, random_seed=0, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """starting map, then emile's entrypoint(). returns (poses, points, start, image_names)"""
    if INCREMENTAL_FOLDER not in sys.path:
        sys.path.insert(0, INCREMENTAL_FOLDER)
    import entrypoint as incremental

    data = load_for_incremental(scene, calibration_path, features_folder)
    start = initialise_reconstruction(data["K"], data["kpts"], data["matches"], random_seed)
    poses, points = incremental.entrypoint(data, start)
    return poses, points, start, data["image_names"]


def describe(scene, start, image_names):
    results = start.all_pair_results.values()
    labels = {label: sum(result.label == label for result in results)
              for label in ("general", "planar", "panoramic", "rejected")}
    seed = start.seed
    return (f"{scene}: {len(image_names)} images, {len(start.all_pair_results)} pairs {labels}\n"
            f"  seed {image_names[seed.first_image]} and {image_names[seed.second_image]} "
            f"(rule {start.seed_choice.rule}), {seed.number_of_points} points, "
            f"median angle {seed.median_triangulation_angle_degrees:.1f} deg")


def main():
    parser = argparse.ArgumentParser(description="build the starting map from the correspondence search results")
    parser.add_argument("scenes", nargs="*", default=SCENES)
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--incremental", action="store_true",
                        help="also run emile's code/icr")
    arguments = parser.parse_args()
    for scene in arguments.scenes:
        if arguments.incremental:
            poses, points, start, image_names = run_incremental(scene, arguments.random_seed, arguments.calibration)
            print(describe(scene, start, image_names), flush=True)
            missing = [image_names[i] for i in range(len(image_names)) if i not in poses]
            print(f"  incremental: {len(poses)} of {len(image_names)} cameras, {len(points)} points, "
                  f"missing {missing}", flush=True)
        else:
            start, image_names = initialise_scene(scene, arguments.random_seed, arguments.calibration)
            print(describe(scene, start, image_names), flush=True)


if __name__ == "__main__":
    main()
