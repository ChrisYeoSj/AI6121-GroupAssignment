import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config as C
from .pipeline import process_scene

COLORS = {"general": "tab:green", "planar": "tab:orange", "panoramic": "tab:blue"}


def row(scene, tag):
    d = os.path.join(C.OUT, scene, tag)
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    with open(os.path.join(d, "pairs.json")) as f:
        pj = json.load(f)["pairs"]
    n = s["n_images"]
    use = [(k, g) for k, g in pj.items() if g["label"] in C.USABLE]
    gap = lambda k: (lambda i, j: min(j - i, n - (j - i)))(*map(int, k.split("-")))
    L = s["labels"]
    return dict(scene=scene, variant=tag, good=L["general"], flat=L["planar"], rot=L["panoramic"],
                bad=L["rejected"], usable=len(use),
                far=sum(gap(k) > n // 4 for k, _ in use),
                putative=round(float(np.mean([g["n_putative"] for g in pj.values()])), 1),
                inliers=round(float(np.mean([g["n_inliers"] for _, g in use])), 1) if use else 0,
                tracks=s["n_tracks"], ge9=s["tracks_ge9"], conflicts=s["conflicted"],
                connected=s["connected"], pair_s=round(s["timing_seconds"]["pairs"]))


def run_experiments(sweep_scene, workers=None):
    for s in C.SCENES:
        process_scene(s, "sift", workers=workers)
    for r in C.RATIOS:
        if r != C.MAX_RATIO:
            process_scene(sweep_scene, "rootsift", ratio=r, tag=f"rootsift_r{int(round(r * 100))}", workers=workers)


def write_table():
    rows = [row(*p.split(os.sep)[-3:-1]) for p in sorted(glob.glob(os.path.join(C.OUT, "*", "*", "summary.json")))]
    rows.insert(0, {h: h for h in rows[0]})
    w = {h: max(len(str(r[h])) for r in rows) for h in rows[0]}
    out = "\n".join("  ".join(str(r[h]).ljust(w[h]) for h in w) for r in rows)
    os.makedirs(os.path.join(C.OUT, "experiments"), exist_ok=True)
    with open(os.path.join(C.OUT, "experiments", "table.txt"), "w") as f:
        f.write(out + "\n")
    return out


def draw(scene, tag):
    d = os.path.join(C.OUT, scene, tag)
    with open(os.path.join(d, "pairs.json")) as f:
        pj = json.load(f)
    n = len(pj["image_names"])
    M = np.zeros((n, n))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    for key, g in pj["pairs"].items():
        i, j = map(int, key.split("-"))
        if g["label"] in C.USABLE:
            M[i, j] = M[j, i] = g["n_inliers"]
            ax[0].plot(*zip(xy[i], xy[j]), color=COLORS[g["label"]], lw=0.3 + g["n_inliers"] / 400, alpha=0.5)
    ax[0].scatter(*xy.T, s=14, c="k", zorder=3)
    for k in range(n):
        ax[0].text(*(xy[k] * 1.07), str(k + 1), ha="center", va="center", fontsize=6)
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title(f"{scene} [{tag}] view graph")
    im = ax[1].imshow(np.log1p(M), cmap="viridis")
    ax[1].set_title("log(1+inliers)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    os.makedirs(os.path.join(C.OUT, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT, "figures", f"{scene}_{tag}_viewgraph.png"), dpi=130)
    plt.close(fig)
