# two_view_opencv.py

This builds the starting 3D map that incremental reconstruction grows from.
Run it with `.venv/bin/python` (needs numpy 2 and OpenCV).

## Usage

```python
import sys
sys.path.insert(0, "scripts/geometry")
from two_view_opencv import initialise_reconstruction

start = initialise_reconstruction(camera_matrix, keypoints_by_image, matches_by_pair, random_seed=0)
```

What it needs:

- `camera_matrix`: the 3x3 K, shared by all images.
- `keypoints_by_image`: `{image_id: (N, 2)}` pixel coordinates. Undistort them first.
- `matches_by_pair`: `{(i, j): (M, 2)}` numpy array. Row `[a, b]` means keypoint `a` in image `i` matches keypoint `b` in image `j`.

The same `random_seed` always gives the same answer.

## What comes out

- `start.poses`: `{image_id: (rotation, translation)}` for the two seed images.
- `start.points`: the (P, 3) 3D points.
- `start.observations`: `{image_id: (P,)}`. Point `k` was seen at keypoint `observations[i][k]` in seed image `i`.
- `start.seed`: the seed pair's result.
- `start.all_pair_results`: `{(i, j): TwoViewResult}` for every pair.

the poses follow `X_camera = rotation @ X_world + translation`.
the first seed camera is the world frame, so its pose is `(I, 0)`.
the second camera's translation has length 1. Two views can't tell the real scale, so the whole map is in these units.

If no pair is good enough to start from, it raises an exception `NoSeedPairFound`.

## Which pairs to use later

every pair in `start.all_pair_results` has a `label`, and `reason` states why

- `general`: has a pose and 3D points. Use it.
- `planar`: the scene is close to one flat surface. no pose, but the matches are fine to use.
- `panoramic`: the camera turned but barely moved. no pose. the matches still help find a new camera's pose, but don't make new points from this pair alone.
- `rejected`: skip it.

## Helpers to reuse

- `calulate_pixels_to_normalised_image_coordinates(K, pixels)` turns (N, 2) pixels into (N, 3) normalised image coordinates.
- `pixel_threshold_to_normalised_units(K, pixels)` turns a pixel threshold into normalised units.
- `find_valid_points(K, rotations, translations, normalised_per_camera, points)` says which points to keep. A point must be in front of every camera, reproject within 3 px, and be seen from at least 2° apart. It works for any number of cameras.
- `triangulation_angles_degrees(rotations, translations, points)` gives the widest viewing angle at each point, for any number of cameras.
- `essential_matrix_errors` and `homography_errors` give the Sampson distance of each match to E or H.
- `INLIER_THRESHOLD_PIXELS` (3 px) and `MINIMUM_TRIANGULATION_ANGLE_DEGREES` (2°). Use the same values so both stages agree.

Note: `triangulate_two_views` only works for two cameras with the first at `(I, 0)`.
For a new camera, call `cv2.triangulatePoints` with both cameras' real `[R | t]`.

## Running it on the correspondence search results

`initialise_from_features.py` loads Asmita's results for a scene (through `scripts/correspondence_search/load.py`), removes the lens distortion from the keypoints, and builds the starting map.

```
python scripts/geometry/initialise_from_features.py --calibration path/to/calibration.json
python scripts/geometry/initialise_from_features.py yunnan_rock_1 --calibration path/to/calibration.json
```

`--calibration` defaults to `results/calibration/calibration.json`, which comes from the calibration stage and is not on this branch yet.

```python
from initialise_from_features import initialise_scene
start, image_names = initialise_scene("yunnan_rock_1", calibration_path="path/to/calibration.json")
```
