import json
import os
import time
from multiprocessing import Pool
import numpy as np
import config as C
from . import features as Fe
from .verification import two_view


G = {}


def init(scene, norm, params):
    C.MIN_NUM_INLIERS = params["min_inliers"]
    K, _ = Fe.load_calibration(scene)
    data = [Fe.load_image_features(scene, nm, C.NORMS[norm]) for nm in Fe.image_names(scene)]
    G.update(K=K, data=data, params=params)


def work(ij):
    i, j = ij
    (k1, d1), (k2, d2) = G["data"][i], G["data"][j]
    p = G["params"]
    m = Fe.match(d1, d2, p["ratio"], p["max_distance"])
    g = two_view(k1[m[:, 0]], k2[m[:, 1]], G["K"], np.random.default_rng(i * 100003 + j))
    mask = g.pop("mask")
    inl = m[mask] if mask is not None else np.zeros((0, 2), np.int32)
    g["n_inliers"] = int(len(inl))
    return i, j, g, inl


def process_scene(scene, norm="rootsift", ratio=C.MAX_RATIO, max_distance=C.MAX_DISTANCE,
                  min_inliers=C.MIN_NUM_INLIERS, tag=None, workers=None):
    tag = tag or norm
    out = os.path.join(C.OUT, scene, tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    names = Fe.extract_scene(scene)
    n = len(names)
    params = dict(ratio=ratio, max_distance=max_distance, min_inliers=min_inliers)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t1 = time.time()
    with Pool(workers or min(os.cpu_count() or 1, 8), initializer=init, initargs=(scene, norm, params)) as pool:
        results = list(pool.imap_unordered(work, pairs, chunksize=4))
    t2 = time.time()
    results.sort(key=lambda r: (r[0], r[1]))
    pj, matches, usable = {}, {}, {}
    for i, j, g, inl in results:
        g["label_name"] = C.LABELS[g["label"]]
        pj[f"{i}-{j}"] = g
        if g["label"] in C.USABLE and len(inl):
            matches[f"m_{i:03d}_{j:03d}"] = inl
            usable[(i, j)] = inl
    with open(os.path.join(out, "pairs.json"), "w") as f:
        json.dump(dict(scene=scene, variant=tag, image_names=names, label_legend=C.LABELS, params=params, pairs=pj), f)
    np.savez_compressed(os.path.join(out, "matches.npz"), **matches)
    n_kpts = []
    for nm in names:
        with np.load(Fe.feature_path(scene, nm)) as z:
            n_kpts.append(len(z["kpts"]))
    obs, lens, conflicts = build_tracks(n_kpts, usable)
    np.savez_compressed(os.path.join(out, "tracks.npz"), observations=obs, track_lengths=lens,
                        image_names=np.array(names), n_kpts=np.array(n_kpts, np.int32))
    labels = {k: sum(1 for g in pj.values() if g["label"] == k) for k in C.LABELS}
    comp = components(n, list(usable))
    summary = dict(scene=scene, variant=tag, norm=norm, params=params, n_images=n, n_pairs=len(pairs),
                   labels=labels, usable_edges=len(usable), connected=len(set(comp)) == 1,
                   n_tracks=int(len(lens)), mean_track_len=float(lens.mean()) if len(lens) else 0.0,
                   max_track_len=int(lens.max()) if len(lens) else 0, tracks_ge9=int((lens >= 9).sum()),
                   conflicted=conflicts,
                   timing_seconds=dict(features=t1 - t0, pairs=t2 - t1, total=time.time() - t0))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"  {scene} [{tag}] {labels} usable={len(usable)} tracks={len(lens)} "
          f">=9:{summary['tracks_ge9']} conflicts={conflicts} connected={summary['connected']} "
          f"pairs {t2 - t1:.0f}s", flush=True)
    return summary


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(x) for x in range(n)]


def build_tracks(n_kpts, pair_matches):
    off = np.concatenate([[0], np.cumsum(n_kpts)])
    edges = []
    for (i, j), m in pair_matches.items():
        edges.extend(zip((off[i] + m[:, 0]).tolist(), (off[j] + m[:, 1]).tolist()))
    root = components(int(off[-1]), edges)
    groups = {}
    for x in sorted({x for e in edges for x in e}):
        groups.setdefault(root[x], []).append(x)
    obs, lens, conflicts = [], [], 0
    for members in groups.values():
        imgs = np.searchsorted(off, members, side="right") - 1
        if len(set(imgs.tolist())) != len(members):
            conflicts += 1
            continue
        tid = len(lens)
        for x, im in sorted(zip(members, imgs.tolist()), key=lambda t: t[1]):
            obs.append((tid, im, x - int(off[im])))
        lens.append(len(members))
    return np.array(obs, np.int32).reshape(-1, 3), np.array(lens, np.int32), conflicts
