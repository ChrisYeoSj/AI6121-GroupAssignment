#!/usr/bin/env python3
"""
make_yup.py - Build the viewing copy of a COLMAP model: gravity = +Y.

run_colmap.py calls this automatically after mapping (unless --no-yup). Run it
by hand for a workspace made with --no-yup or before that became the default.

For a workspace written by run_colmap.py (results/colmap_out/<scene>/sparse/0)
this produces the same layout as results/colmap_out/yunnan_rock:

    sparse/0_raw/                 COLMAP's own gauge (sparse/0, renamed).
                                  Quote numbers from here and benchmark.json.
    sparse/0_yup/                 same model, rotated + recentred. Viewing only.
    <scene>_yup2.ply              points in the yup gauge (binary)
    <scene>_with_cameras.ply      the same points + one red vertex per camera (ascii)
    preview.jpg                   top-down and side views of the yup model

The transform is rigid (no scale), so every benchmark number is unchanged:
    origin  = centroid of the camera centres
    +Y      = normal of the plane best fitting the camera centres, signed so the
              scene lies on the +Y side (OpenCV convention: +Y points down)
    R       = the smallest rotation taking that normal onto +Y
This assumes the photos were taken on a roughly level ring around the object,
which is how every scene in this project is shot.

Usage:
    python make_yup.py results/colmap_out/yunnan_rock_2
    python make_yup.py results/colmap_out/yunnan_rock_2 results/colmap_out/yunnan_rock_2_lower_height
"""

import os
import struct
import sys

import numpy as np
import pycolmap


def yup_transform(rec):
    """Return (R, c0): x_yup = R @ (x_raw - c0)."""
    centres = np.array([im.projection_center() for im in rec.images.values()])
    c0 = centres.mean(0)
    n = np.linalg.svd(centres - c0)[2][2]
    pts = np.array([p.xyz for p in rec.points3D.values()])
    if np.median((pts - c0) @ n) < 0:
        n = -n
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(n, y), n @ y
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K / (1 + c)
    return R, c0


def write_ply_binary(path, xyz, rgb):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    with open(path, "wb") as fh:
        fh.write(header)
        for p, c in zip(xyz, rgb):
            fh.write(struct.pack("<fffBBB", *p, *c))


def write_ply_ascii(path, xyz, rgb):
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n"
                 f"element vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n")
        for p, c in zip(xyz, rgb):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def write_preview(path, xyz, rgb, cams, panel=600):
    """Two 600x600 panels on a dark ground: top-down (x-z) and side (x-y, +Y down)."""
    import cv2

    # frame both views on the camera ring; far background points fall outside
    r = np.abs(cams[:, [0, 2]]).max() * 1.15
    scale = (panel / 2 - 20) / r
    img = np.full((panel, 2 * panel, 3), 17, np.uint8)
    order = np.argsort(-xyz[:, 1])              # far-below first, so upper points draw on top

    def draw(ox, u, v, pts, cols, radius):
        px = np.round(ox + panel / 2 + u * scale).astype(int)
        py = np.round(panel / 2 + v * scale).astype(int)
        ok = (px >= ox) & (px < ox + panel) & (py >= 30) & (py < panel)
        for x, y, c in zip(px[ok], py[ok], cols[ok]):
            cv2.circle(img, (int(x), int(y)), radius, tuple(int(k) for k in c[::-1]), -1)

    p, col = xyz[order], rgb[order]
    red = np.tile([255, 0, 0], (len(cams), 1))
    draw(0, p[:, 0], p[:, 2], p, col, 1)                        # top-down: x right, z down
    draw(0, cams[:, 0], cams[:, 2], cams, red, 4)
    y_mid = (cams[:, 1].mean() + np.percentile(xyz[:, 1], 90)) / 2
    draw(panel, p[:, 0], p[:, 1] - y_mid, p, col, 1)            # side: +Y down the page
    draw(panel, cams[:, 0], cams[:, 1] - y_mid, cams, red, 4)
    for ox, text in ((0, f"TOP-DOWN (x-z): {len(cams)} cameras"),
                     (panel, "SIDE (x-y, +Y down)")):
        cv2.putText(img, text, (ox + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process(workspace):
    scene = os.path.basename(os.path.normpath(workspace))
    sparse = os.path.join(workspace, "sparse")
    raw_dir = os.path.join(sparse, "0_raw")
    yup_dir = os.path.join(sparse, "0_yup")

    if os.path.isdir(yup_dir):
        raise RuntimeError(f"{yup_dir} already exists - remove it first to rebuild.")
    if not os.path.isdir(raw_dir):
        models = [d for d in os.listdir(sparse)
                  if os.path.exists(os.path.join(sparse, d, "images.bin"))]
        if models != ["0"]:
            raise RuntimeError(f"{sparse}: expected exactly one model '0', found "
                               f"{models}. Pick one by hand before running this.")
        os.rename(os.path.join(sparse, "0"), raw_dir)
        print(f"  sparse/0 -> sparse/0_raw")

    rec = pycolmap.Reconstruction(raw_dir)
    R, c0 = yup_transform(rec)
    rec.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(R), -R @ c0))
    os.makedirs(yup_dir)
    rec.write(yup_dir)
    rec = pycolmap.Reconstruction(yup_dir)      # PLYs list cameras in on-disk order

    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts])        # float64; the ascii PLY rounds from these
    rgb = np.array([p.color for p in pts], dtype=np.uint8)
    cams = np.array([im.projection_center() for im in rec.images.values()])

    write_ply_binary(os.path.join(workspace, f"{scene}_yup2.ply"),
                     xyz.astype(np.float32), rgb)
    write_ply_ascii(os.path.join(workspace, f"{scene}_with_cameras.ply"),
                    np.vstack([xyz, cams]),
                    np.vstack([rgb, np.tile([255, 0, 0], (len(cams), 1))]).astype(int))
    write_preview(os.path.join(workspace, "preview.jpg"), xyz, rgb, cams)

    heights = cams[:, 1]
    print(f"  0_yup written: {len(rec.images)} cameras, {len(pts)} points; "
          f"camera height spread {heights.std():.3f} (should be small for a level ring)")
    print(f"  {scene}_yup2.ply, {scene}_with_cameras.ply, preview.jpg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for ws in sys.argv[1:]:
        print(ws)
        try:
            process(ws)
        except RuntimeError as e:
            sys.exit(str(e))
