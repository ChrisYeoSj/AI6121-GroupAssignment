import json
import os
import numpy as np
import config as C
from utils import features as Fe


def load(scene, variant="rootsift"):
    d = os.path.join(C.OUT, scene, variant)
    with open(os.path.join(d, "pairs.json")) as f:
        pj = json.load(f)
    names = pj["image_names"]
    kpts = []
    for nm in names:
        with np.load(Fe.feature_path(scene, nm)) as z:
            kpts.append(z["kpts"].astype(np.float64))
    K, dist = Fe.load_calibration(scene)
    with np.load(os.path.join(d, "matches.npz")) as z:
        matches = {tuple(int(v) for v in k.split("_")[1:]): z[k] for k in z.files}
    with np.load(os.path.join(d, "tracks.npz")) as z:
        obs = z["observations"]
    tracks = np.split(obs[:, 1:], np.cumsum(np.bincount(obs[:, 0]))[:-1]) if len(obs) else []
    pairs = {tuple(int(v) for v in k.split("-")): g for k, g in pj["pairs"].items()}
    return dict(image_names=names, K=K, dist=dist, kpts=kpts, matches=matches, pairs=pairs, tracks=tracks)
