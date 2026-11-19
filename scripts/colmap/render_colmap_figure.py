"""Render COLMAP text models as print-friendly figures (white background, gravity up).

usage: python render_colmap.py <txt_dir> <out.png> [azim] [elev]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


def read_points(path):
    xyz, rgb, err = [], [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        err.append(float(p[7]))
    return np.array(xyz), np.array(rgb) / 255.0, np.array(err)


def read_cams(path):
    lines = [l for l in open(path) if not l.startswith("#")]
    Rs, Cs = [], []
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        R = qvec2rot([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        Rs.append(R)
        Cs.append(-R.T @ t)
    return np.array(Rs), np.array(Cs)


def main():
    d, out = sys.argv[1], sys.argv[2]
    azim_auto = len(sys.argv) <= 3 or sys.argv[3] == "auto"
    azim = 0.0 if azim_auto else float(sys.argv[3])
    azim_off = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    elev = float(sys.argv[4]) if len(sys.argv) > 4 else 22
    xyz, rgb, err = read_points(f"{d}/points3D.txt")
    Rs, Cs = read_cams(f"{d}/images.txt")

    # Gravity: normal of the plane through the camera ring; ground lies below it.
    c0 = Cs.mean(0)
    up = np.linalg.svd(Cs - c0)[2][2]
    if np.median((xyz - c0) @ up) > 0:
        up = -up
    ex = np.cross(up, [1, 0, 0]); ex /= np.linalg.norm(ex)
    ey = np.cross(up, ex)
    B = np.stack([ex, ey, up])                      # world -> (x, y, up)
    ring_r = np.median(np.linalg.norm((Cs - c0) @ B[:2].T, axis=1))

    P = (xyz - c0) @ B.T
    C = (Cs - c0) @ B.T
    rad = np.linalg.norm(P[:, :2], axis=1)
    keep = (rad < 1.15 * ring_r) & (err < 2.0)
    z_ground = np.percentile(P[keep, 2], 5)
    keep &= (P[:, 2] > z_ground - 0.15 * ring_r) & (P[:, 2] < C[:, 2].max() + 0.5 * ring_r)
    P, col = P[keep], rgb[keep]
    # The rock: points well inside the ring; draw them larger than the ground.
    rock = np.linalg.norm(P[:, :2], axis=1) < 0.45 * ring_r
    if azim_auto:
        # Look along the rock's thinnest horizontal direction, i.e. at its broad side.
        q = P[rock, :2] - P[rock, :2].mean(0)
        minor = np.linalg.svd(q, full_matrices=False)[2][1]
        azim = np.degrees(np.arctan2(minor[1], minor[0])) + azim_off

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*P[~rock].T, c=col[~rock], s=0.15, linewidths=0, depthshade=False)
    ax.scatter(*P[rock].T, c=col[rock], s=0.5, linewidths=0, depthshade=False)

    # Cameras as small frustums (raw images are landscape 4:3).
    s = 0.07 * ring_r
    segs = []
    for R, c in zip(Rs, C):
        Rw = B @ R.T                                 # camera axes in plot frame
        x, y, z = Rw[:, 0], Rw[:, 1], Rw[:, 2]
        corners = [c + s * (z + 0.66 * a * x + 0.5 * b * y) for a, b in
                   [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        for k in range(4):
            segs.append([c, corners[k]])
            segs.append([corners[k], corners[(k + 1) % 4]])
    ax.add_collection3d(Line3DCollection(segs, colors="#d62728", linewidths=0.45))

    allp = np.vstack([P, C])
    lo, hi = allp.min(0), allp.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    crop(out)
    print(out, "points", keep.sum(), "cams", len(C))


def crop(path, pad=12):
    img = plt.imread(path)
    mask = (img[..., :3] < 0.97).any(-1)
    ys, xs = np.where(mask)
    img = img[max(ys.min()-pad, 0):ys.max()+pad, max(xs.min()-pad, 0):xs.max()+pad]
    plt.imsave(path, img)


if __name__ == "__main__":
    main()
