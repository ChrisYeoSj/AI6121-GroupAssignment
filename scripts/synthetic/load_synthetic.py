#!/usr/bin/env python3
"""
load_synthetic.py - load the synthetic test scene in the Part 7 data contract.

    from load_synthetic import load
    S = load("clean")            # or "noisy" / "outliers"

    S.K                 (3,3)  float64   shared intrinsics, never refined
    S.dist              (5,)   float64   ALL ZERO - observations are undistorted
    S.calib_resolution  (w, h)
    S.features[i]    -> (kpts (N,2) float32, desc (N,128) float32, colors (N,3) uint8)
    S.matches[(i,j)] -> (M,2) int32      putative matches, keypoint INDICES

  Ground truth (for asserting against; a real pipeline never sees these):
    S.gt_poses[i]    -> (rvec (3,), tvec (3,))   X_cam = R @ X_world + tvec
    S.gt_centres[i]  -> (3,)                      camera position in world
    S.gt_xyz         (P,3) float64
    S.gt_rgb         (P,3) uint8
    S.gt_kp_point_id[i] -> (N,) int32   3D point index per keypoint, -1 = decoy
    S.gt_pair_label[(i,j)] -> str       general|planar|panoramic|rejected
    S.gt_pair_alpha[(i,j)] -> float     median triangulation angle, degrees
    S.gt_match_inlier[(i,j)] -> (M,) bool  which rows of matches are CORRECT
                                            (all True except in "outliers")
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))        # repo root
VARIANTS = ("clean", "noisy", "outliers")

# Code lives in scripts/synthetic/, data in results/synthetic/. Search both, so
# the loader keeps working whichever way the tree is arranged, and honour an
# explicit override for anyone keeping the data elsewhere.
_SEARCH = [
    os.environ.get("SYNTHETIC_DIR"),
    os.path.join(_ROOT, "results", "synthetic"),
    _HERE,
    os.path.join(_ROOT, "synthetic"),
]


def data_dir():
    """First directory on the search path that actually holds the variants."""
    for d in _SEARCH:
        if d and os.path.isfile(os.path.join(d, "clean.npz")):
            return d
    raise FileNotFoundError(
        "Could not find the synthetic .npz files. Looked in:\n  "
        + "\n  ".join(str(d) for d in _SEARCH if d)
        + "\nSet SYNTHETIC_DIR, or pass load(..., path=...).")


class Scene:
    __slots__ = ("K", "dist", "calib_resolution", "n_cameras", "variant",
                 "features", "matches", "gt_poses", "gt_centres", "gt_xyz",
                 "gt_rgb", "gt_kp_point_id", "gt_pair_label", "gt_pair_alpha",
                 "gt_match_inlier", "camera_kind")

    def image_ids(self):
        return list(range(self.n_cameras))

    def pairs(self, label=None):
        """All pairs, or only those carrying a given ground-truth label."""
        if label is None:
            return sorted(self.matches)
        return sorted(k for k in self.matches if self.gt_pair_label[k] == label)

    def __repr__(self):
        from collections import Counter
        c = Counter(self.gt_pair_label.values())
        return (f"<Scene {self.variant}: {self.n_cameras} cameras, "
                f"{len(self.gt_xyz)} points, {len(self.matches)} pairs {dict(c)}>")


def load(variant="clean", path=None):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    npz = path or os.path.join(data_dir(), f"{variant}.npz")
    d = np.load(npz, allow_pickle=True)

    s = Scene()
    s.variant = variant
    s.K = d["K"].astype(np.float64)
    s.dist = d["dist"].astype(np.float64)
    s.calib_resolution = tuple(int(v) for v in d["calib_resolution"])
    s.n_cameras = int(d["n_cameras"])
    s.camera_kind = [str(x) for x in d["camera_kind"]]

    s.features, s.gt_kp_point_id = {}, {}
    for i in range(s.n_cameras):
        s.features[i] = (d[f"kpts_{i:02d}"].astype(np.float32),
                         d[f"desc_{i:02d}"].astype(np.float32),
                         d[f"colors_{i:02d}"].astype(np.uint8))
        s.gt_kp_point_id[i] = d[f"ptid_{i:02d}"].astype(np.int32)

    s.matches, s.gt_match_inlier = {}, {}
    s.gt_pair_label, s.gt_pair_alpha = {}, {}
    for key in d.files:
        if not key.startswith("m_"):
            continue
        i, j = int(key[2:4]), int(key[5:7])
        s.matches[(i, j)] = d[key].astype(np.int32)
        s.gt_match_inlier[(i, j)] = d[f"minl_{i:02d}_{j:02d}"].astype(bool)
        lab = d[f"lab_{i:02d}_{j:02d}"]
        s.gt_pair_label[(i, j)] = str(lab[0])
        s.gt_pair_alpha[(i, j)] = float(lab[1])

    rv, tv = d["gt_rvecs"], d["gt_tvecs"]
    s.gt_poses = {i: (rv[i].astype(np.float64), tv[i].astype(np.float64))
                  for i in range(s.n_cameras)}
    s.gt_centres = {i: d["gt_centres"][i].astype(np.float64)
                    for i in range(s.n_cameras)}
    s.gt_xyz = d["gt_xyz"].astype(np.float64)
    s.gt_rgb = d["gt_rgb"].astype(np.uint8)
    return s


if __name__ == "__main__":
    for v in VARIANTS:
        print(load(v))
