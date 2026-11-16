"""
Integration with Features (Asmita's Code)
Load Asmita's correspondence search results for one scene and build the
starting map with two_view_opencv.py.

`python scripts/geometry/initialise_from_features.py [scenes] [--calibration PATH] [--random-seed N]`

it will run on all 4 rock scenes, if no scenes are given,

This file is the only place that knows about the correspondence search folders.
two_view_opencv.py still takes plain arrays, so it works with any source of matches.

What it does for each scene:
    1. loads the keypoints, matches and calibration with asmita's load.py
       (scripts/correspondence_search/load.py).
    2. removes the lens distortion from every keypoint. Her keypoints are raw
       pixels, but two_view_opencv.py expects undistorted ones.
    3. runs initialise_reconstruction() on the result.

Asmita's keypoints are in the upright photo (cv2.imread turns the photo upright), and
her K is for the upright photo too, so the two match and nothing needs turning

depends on results/calibration/calibration.json from the calibration stage. It is not
on the branch, run --calibration to point at a copy
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

from load import load  # asmita's loader, needs only numpy
from two_view_opencv import initialise_reconstruction # chris's code

SCENES = ["yunnan_rock_1", "yunnan_rock_1_lower_height", "yunnan_rock_2", "yunnan_rock_2_lower_height"]
FEATURES_FOLDER = os.path.join(_ROOT, "results", "correspondence_search")
CALIBRATION_PATH = os.path.join(_ROOT, "results", "calibration", "calibration.json")

# cv2.undistortPoints finds each undistorted pixel by repeated guessing. its a default
# of 5 rounds, which is not enough for this lens near the corners, so we ask for more.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)


def undistort_keypoints(camera_matrix, distortion, keypoints):
    """raw pixels (N,2) -> undistorted pixels (N,2), with the same K."""
    if len(keypoints) == 0:
        return np.zeros((0, 2))
    undistorted = cv2.undistortPoints(np.asarray(keypoints, dtype=np.float64).reshape(-1, 1, 2),
                                      camera_matrix, distortion, P=camera_matrix,
                                      criteria=UNDISTORT_CRITERIA)
    return undistorted.reshape(-1, 2)


def load_features(scene, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """one scene's correspondence search results, ready for two_view_opencv.py.

    returns (camera_matrix, keypoints_by_image, matches_by_pair, image_names).
    image ids are positions in image_names, the same as in her results
    """
    if not os.path.isfile(calibration_path):
        raise FileNotFoundError(f"no calibration at {calibration_path}. Pass --calibration.")
    loaded = load(scene, out=features_folder, calib=calibration_path)
    camera_matrix, distortion = loaded["K"], loaded["dist"]
    keypoints_by_image = {image: undistort_keypoints(camera_matrix, distortion, keypoints)
                          for image, keypoints in enumerate(loaded["kpts"])}
    matches_by_pair = {pair: np.asarray(matches, dtype=np.int64) for pair, matches in loaded["matches"].items()}
    return camera_matrix, keypoints_by_image, matches_by_pair, loaded["image_names"]


def initialise_scene(scene, random_seed=0, calibration_path=CALIBRATION_PATH, features_folder=FEATURES_FOLDER):
    """load one scene and build its starting map. returns (InitialReconstruction, image_names)."""
    camera_matrix, keypoints_by_image, matches_by_pair, image_names = load_features(
        scene, calibration_path, features_folder)
    start = initialise_reconstruction(camera_matrix, keypoints_by_image, matches_by_pair, random_seed)
    return start, image_names


def describe(scene, start, image_names):
    """a few plain lines about one scene's starting map."""
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
    arguments = parser.parse_args()
    for scene in arguments.scenes:
        start, image_names = initialise_scene(scene, arguments.random_seed, arguments.calibration)
        print(describe(scene, start, image_names), flush=True)


if __name__ == "__main__":
    main()
