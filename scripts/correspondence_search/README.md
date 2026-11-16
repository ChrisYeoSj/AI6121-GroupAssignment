# Correspondence search (stage 2 for AI6121)

Local only (listed in `.git/info/exclude`). Where each piece comes from in COLMAP is in `REFERENCE.md`.

## Layout

```
correspondence_search/
  main.py          entry point (run | experiments | graph | load | all)
  config.py        paths, scenes, all thresholds
  load.py          loader for Teammate 3 ;-;
  utils/
    features.py    SIFT detection, RootSIFT / SIFT normalisation, per-image cache, calibration, unit rays,
                   brute-force matching (ratio test, distance cap, cross check)
    geometry.py    7-point F, 8-point F, 4-point H, LO-RANSAC with MSAC, F vs H decision,
                   pose from E or H, triangulation angle, pair label
    pipeline.py    process_scene (every image pair of a scene in parallel, writes results), tracks
    analysis.py    SIFT and ratio-sweep experiments, comparison table, view-graph figures
```

Requires: Python 3, numpy, opencv-python, matplotlib. Stage 1 `results/calibration/calibration.json` and `photos/<scene>/` must be present.

## Run

Works from any directory:

```
python scripts/correspondence_search/main.py all
python scripts/correspondence_search/main.py run
python scripts/correspondence_search/main.py run yunnan_rock_1 --norm sift --ratio 0.9 --tag test
python scripts/correspondence_search/main.py experiments --run
python scripts/correspondence_search/main.py experiments
python scripts/correspondence_search/main.py graph
python scripts/correspondence_search/main.py load
```

| Command | Does |
|---|---|
| `all` | everything below, in order (about 1.5 h on 8 cores, mostly the ratio 0.9 / 1.0 runs) |
| `run [scenes]` | main pipeline (all scenes if none given), default RootSIFT ratio 0.8, about 1.5 min per scene once features are cached |
| `experiments --run` | SIFT on all scenes plus the ratio sweep, then the table |
| `experiments` | table only, from existing results |
| `graph` | view-graph figures |
| `load` | checks that every scene loads |

`--workers N` sets the process count (default min(cores, 8)).

## Outputs: `results/correspondence_search/`

- `<scene>/features/img_XXX.npz`: `kpts` (N,2) pixel coordinates in the original image, `desc_raw` (N,128) uint8, `wh`, `key`. If `key` matches the photo's size, mtime and parameters, the file is reused and nothing is recomputed.
- `<scene>/<variant>/pairs.json`: one entry per image pair `"i-j"` with `label`, `label_name`, `config`, `n_putative`, `n_f`, `n_h`, `h_f`, `n_inliers`, `angle`, `reason`.
- `<scene>/<variant>/matches.npz`: `m_iii_jjj` (M,2) keypoint indices of the verified inlier matches, only for usable pairs.
- `<scene>/<variant>/tracks.npz`: `observations` (T,3) rows `[track_id, image_index, keypoint_index]`, `track_lengths`, `image_names`, `n_kpts`.
- `<scene>/<variant>/summary.json`: label counts, usable edges, connectivity, track statistics, timings.
- `figures/<scene>_rootsift_viewgraph.png`: view graph.
- `experiments/table.txt`: all variants side by side.

Variants: `rootsift` (default, ratio 0.8), `sift`, `rootsift_r60`, `rootsift_r70`, `rootsift_r90`, `rootsift_r100` (ratio 1.0 = no ratio test).

## Pair labels

| label | COLMAP configuration | meaning 
|---|---|---
| good | UNCALIBRATED | F explains the matches, enough parallax 
| flat | PLANAR | H explains nearly as many inliers as F, and the H decomposition gives a translation 
| rotation-only | PANORAMIC, or median triangulation angle < 1.5 deg | no usable baseline 
| suspicious | DEGENERATE | fewer than 15 putative matches or verified inliers 

Usable pairs (written to `matches.npz` and used for tracks): good, flat, rotation-only.

## Instructions for Teammate 3 ;-;

```python
import sys; sys.path.insert(0, "scripts/correspondence_search")
from load import load
r = load("yunnan_rock_1")
r["K"], r["dist"], r["kpts"][i], r["matches"][(i, j)], r["pairs"][(i, j)]["label"], r["tracks"][t]
```

`r["tracks"][t]` is an array of `[image_index, keypoint_index]` rows. Every track has at most one keypoint per image.
