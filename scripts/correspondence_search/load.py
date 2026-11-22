import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "correspondence_search")
CALIB = os.path.join(ROOT, "results", "calibration", "calibration.json")


def calibration(w, h, calib=CALIB):
    with open(calib) as f:
        c = json.load(f)
    K = np.array(c["K"], np.float64).reshape(3, 3)
    dist = np.array(c.get("dist", [0] * 5), np.float64).ravel()
    cw, ch = c.get("resolution") or c.get("calib_resolution")
    if (cw, ch) != (w, h):
        sx, sy = w / cw, h / ch
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K, dist


def load(scene, variant="rootsift", out=OUT, calib=CALIB):
    d = os.path.join(out, scene, variant)
    with open(os.path.join(d, "pairs.json")) as f:
        pj = json.load(f)
    names = pj["image_names"]
    kpts = []
    for nm in names:
        with np.load(os.path.join(out, scene, "features", os.path.splitext(nm)[0] + ".npz")) as z:
            kpts.append(z["kpts"].astype(np.float64))
            wh = [int(v) for v in z["wh"]]
    K, dist = calibration(*wh, calib)
    with np.load(os.path.join(d, "matches.npz")) as z:
        matches = {tuple(int(v) for v in k.split("_")[1:]): z[k] for k in z.files}
    with np.load(os.path.join(d, "tracks.npz")) as z:
        obs = z["observations"]
    tracks = np.split(obs[:, 1:], np.cumsum(np.bincount(obs[:, 0]))[:-1]) if len(obs) else []
    pairs = {tuple(int(v) for v in k.split("-")): g for k, g in pj["pairs"].items()}
    return dict(image_names=names, K=K, dist=dist, kpts=kpts, matches=matches, pairs=pairs, tracks=tracks)
