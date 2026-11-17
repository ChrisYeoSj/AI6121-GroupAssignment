#!/usr/bin/env python3
"""
calibrate.py — Stage 1 of the AI6121 SfM project: solve for K.

Zhang's method via cv2.calibrateCamera.
Reference: Z. Zhang, "A Flexible New Technique for Camera Calibration",
IEEE TPAMI 22(11):1330-1334, 2000.

    python calibrate.py photos/calib --width 9 --height 6 --square-size 25

--width/--height are INNER CORNERS, not squares. A board with 10x7 squares
has 9x6 inner corners.

Writes calibration.json, which check_photos.py, grade_calibration.py and the
pipeline all read. The 'resolution' field is not optional: a K without the
resolution it was solved at is meaningless, since fx scales with image width.

This script only computes K. Whether the SHOOT was good enough for K to be
trustworthy is a separate question -- run grade_calibration.py for that.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
DETECTORS = ("classic", "sb")


def detect_corners(gray, pattern, detector="classic"):
    """THE corner detector. grade_calibration.py imports this one.

    Both scripts must see the same corners or the grader is judging different
    observations than the calibration used -- that divergence is exactly what
    produced spurious outlier warnings before.

    "classic"  findChessboardCorners + cornerSubPix. Robust, well understood.
    "sb"       findChessboardCornersSB. Finds boards classic misses and is
               ~2% tighter on average, but occasionally puts a single corner
               40-60 px out, which plain least squares has no defence against.

    flags= is a KEYWORD: the third positional parameter of
    findChessboardCorners is `corners`, not `flags`, so passing them
    positionally silently does nothing.
    """
    if detector == "sb":
        ok, c = cv2.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
        return c if ok else None
    ok, c = cv2.findChessboardCorners(gray, pattern, flags=FIND_FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), CRITERIA)


def calibrate(dirpath, pattern, image_format, square_size, prefix="",
              long_side=None, draw_dir=None, detector="classic"):
    """Returns (rms, K, dist, rvecs, tvecs, shape, used_files, per_view_rms)."""
    width, height = pattern

    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    objpoints, imgpoints, used = [], [], []
    shape = None                       # (w, h) -- set once, never from the last file

    files = sorted(glob.glob(os.path.join(dirpath, f"{prefix}*.{image_format}")))
    if not files:
        raise SystemExit(f"no {image_format} files in {dirpath}")

    for fname in files:
        img = cv2.imread(fname)
        if img is None:
            print(f"  skip {os.path.basename(fname)}: unreadable")
            continue

        if long_side:                  # optional: calibrate at processing resolution
            h, w = img.shape[:2]
            sc = long_side / float(max(h, w))
            if sc < 1.0:
                img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))),
                                 interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if shape is None:
            shape = gray.shape[::-1]
        elif gray.shape[::-1] != shape:
            print(f"  skip {os.path.basename(fname)}: size {gray.shape[::-1]} "
                  f"!= {shape}")
            continue

        corners = detect_corners(gray, pattern, detector)
        if corners is None:
            print(f"  skip {os.path.basename(fname)}: no board found")
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(os.path.basename(fname))

        if draw_dir:
            os.makedirs(draw_dir, exist_ok=True)
            vis = cv2.drawChessboardCorners(img.copy(), pattern, corners, True)
            cv2.imwrite(os.path.join(draw_dir, os.path.basename(fname)), vis)

    if not imgpoints:
        raise SystemExit(
            "no chessboard detected in any image.\n"
            "  1. --width/--height count INNER CORNERS, not squares\n"
            "  2. every inner corner must be visible; nothing cropped\n"
            "  3. check for glare on the print and motion blur")

    print(f"  using {len(imgpoints)}/{len(files)} images at {shape[0]}x{shape[1]}")

    # NOTE: the first return value is the RMS reprojection error in PIXELS,
    # not a success flag. It is the headline quality number.
    #
    # The Extended variant also returns the per-view RMS, in the same pixel
    # units, so a bad headline number can be traced to the frames that caused
    # it. Do NOT hand-roll this as norm(pts, proj) / len(pts): cv2.norm returns
    # sqrt(sum of squares), so an RMS divides by sqrt(N), not N. The /N form
    # (from an old OpenCV tutorial) understates the error by sqrt(N) -- a
    # factor of ~7.3 on a 9x6 board -- and turns a 3 px calibration into a
    # reassuring-looking 0.4 px.
    rms, K, dist, rvecs, tvecs, _sd_int, _sd_ext, per_view = \
        cv2.calibrateCameraExtended(objpoints, imgpoints, shape, None, None)
    per_view = np.asarray(per_view, dtype=float).ravel()
    return rms, K, dist, rvecs, tvecs, shape, used, per_view


def main(argv=None):
    ap = argparse.ArgumentParser(description="Camera calibration (Zhang / OpenCV).")
    ap.add_argument("dirpath", help="folder of chessboard photos")
    ap.add_argument("--width", type=int, default=9, help="inner corners across")
    ap.add_argument("--height", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-size", type=float, default=25.0, help="mm")
    ap.add_argument("--format", default="jpg", dest="image_format")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--long-side", type=int, default=None,
                    help="calibrate at this long side (e.g. 1600 to match the "
                         "pipeline's processing resolution). Default: full size.")
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=os.path.join(_root, "results", "calibration",
                                                  "calibration.json"))
    ap.add_argument("--detector", choices=DETECTORS, default="classic",
                    help="corner detector; grade_calibration.py must use the same one")
    ap.add_argument("--draw", default=None,
                    help="folder to write corner-overlay images for inspection")
    a = ap.parse_args(argv)

    print(f"calibrating {a.dirpath}  pattern {a.width}x{a.height} inner corners")
    rms, K, dist, rvecs, tvecs, shape, used, per_view = calibrate(
        a.dirpath, (a.width, a.height), a.image_format, a.square_size,
        a.prefix, a.long_side, a.draw, a.detector)

    print(f"\n  RMS  = {rms:.4f} px  at {shape[0]}x{shape[1]}")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  dist = {np.round(dist.ravel(), 5).tolist()}")

    # Per-image RMS, worst first. The headline RMS is an average: one blurred
    # or flexed board can hide behind twenty good ones. Delete the offenders
    # and re-run -- if the RMS does not drop, the board itself is the problem.
    order = np.argsort(per_view)[::-1]
    print(f"\n  per-image RMS, worst first (median {np.median(per_view):.3f} px):")
    for i in order[:5]:
        print(f"    {per_view[i]:7.3f} px  {used[i]}")
    if len(order) > 5:
        print(f"    ... and {len(order) - 5} more, best {per_view[order[-1]]:.3f} px")

    json.dump({
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "resolution": [int(shape[0]), int(shape[1])],
        "rms": float(rms),
        "n_images": len(used),
        "pattern_inner_corners": [a.width, a.height],
        "square_size_mm": a.square_size,
        "detector": a.detector,
        "images": used,
        "per_view_rms": per_view.tolist(),   # aligned with "images"
    }, open(a.out, "w"), indent=2)
    print(f"\n  wrote {a.out}")
    print(f"  next: python grade_calibration.py {a.dirpath} --calib {a.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
