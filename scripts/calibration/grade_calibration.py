#!/usr/bin/env python3
"""
grade_calibration.py — did I shoot the chessboard correctly?

Reads a calibration.json produced by calibrate.py and the folder it came
from, and answers the question the RMS cannot: is this K trustworthy?

WHY THIS IS A SEPARATE QUESTION
    A low RMS on a bad shoot is the classic calibration trap. Twenty-five
    near-fronto-parallel photos all taken from the same distance will report
    RMS 0.3 px and hand you a focal length wrong by 10%, because nothing in
    that set distinguishes a large board far away from a small board close
    up. The RMS measures how well the model fits the data you gave it. It
    says nothing about whether you gave it the right data.

WHAT IT GRADES
    1. TILT DIVERSITY   are the boards actually at different angles?
    2. FRAME COVERAGE   did corners reach the edges, where distortion lives?
    3. SCALE DIVERSITY  did you vary the distance?
    4. PER-IMAGE ERROR  is one bad frame dragging the whole solve?
    5. SANITY           aspect ratio, principal point, RMS vs resolution

    It does NOT recalibrate. Poses come from solvePnP against the K you
    already solved, so what it grades is exactly the K you will use.

USAGE
    python grade_calibration.py photos/calib --calib calibration.json
    python grade_calibration.py photos/calib --calib calibration.json \\
           --undistort-check photos/easy/img_001.jpg

EXIT CODES
    0 PASS   1 WARN   2 FAIL
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------

PROC_LONG_SIDE = 4032      # px. Native 3024x4032 portrait (plan, Part 1).
RMS_GOOD = 1.25            # px at PROC_LONG_SIDE (= 0.5 px at 1600, scaled by 2.52)
RMS_BAD = 2.5

MIN_IMAGES_OK = 20
MIN_IMAGES_HARD = 12

TILT_MIN_STRENGTH = 0.15   # dimensionless foreshortening; ~0 is fronto-parallel
TILT_MIN_DIVERSITY = 35.0   # deg, median pairwise angle between tilt DIRECTIONS
TILT_TOO_MUCH = 0.85       # beyond this, corner localisation degrades

FOV_MIN_DEG = 25.0         # a phone main camera is 60-80 deg horizontal
FOV_MAX_DEG = 110.0

COVERAGE_GRID = 6
COVERAGE_MIN = 0.60        # fraction of all cells containing a corner
COVERAGE_EDGE_MIN = 0.40   # fraction of BORDER cells containing a corner

SCALE_MIN_RATIO = 2.0      # largest board area / smallest
PER_IMAGE_ERR_MULT = 2.5

# The detector lives in calibrate.py and is IMPORTED, not re-implemented here.
# These two scripts silently diverging is precisely the bug this fixes: the
# grader judged frames the calibration never saw, and SB's occasional 40-60 px
# single-corner miss tripped the outlier check on perfectly good frames.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import detect_corners, DETECTORS       # noqa: E402


def find_corners(gray, pattern, detector="classic"):
    return detect_corners(gray, pattern, detector)


def natural_key(n):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", n)]


class Report:
    def __init__(self):
        self.rows = []

    def ok(self, s, m): self.rows.append(("OK", s, m))
    def warn(self, s, m): self.rows.append(("WARN", s, m))
    def fail(self, s, m): self.rows.append(("FAIL", s, m))
    def info(self, s, m): self.rows.append(("INFO", s, m))

    @property
    def n_fail(self): return sum(1 for l, _, _ in self.rows if l == "FAIL")

    @property
    def n_warn(self): return sum(1 for l, _, _ in self.rows if l == "WARN")

    def code(self): return 2 if self.n_fail else (1 if self.n_warn else 0)

    def render(self):
        w = max((len(s) for _, s, _ in self.rows), default=8)
        out = []
        for lv in ("FAIL", "WARN", "OK", "INFO"):
            for l, s, m in self.rows:
                if l == lv:
                    out.append(f"  [{lv:4s}] {s:<{w}s}  {m}")
        return "\n".join(out)


def hr(t=""):
    bar = "-" * 78
    return f"\n{bar}\n{t}\n{bar}" if t else bar


# ----------------------------------------------------------------------------

def redetect(dirpath, pattern, shape, image_format, rep, detector="classic"):
    """Re-find corners at the resolution the calibration was solved at."""
    files = sorted(glob.glob(os.path.join(dirpath, f"*.{image_format}")),
                   key=lambda p: natural_key(os.path.basename(p)))
    corners_list, names = [], []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != tuple(shape):
            sc = shape[0] / float(w)
            if abs(shape[1] / float(h) - sc) > 1e-3:
                continue                       # different aspect ratio, skip
            img = cv2.resize(img, tuple(shape), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = find_corners(gray, pattern, detector)
        if c is None:
            continue
        corners_list.append(c)
        names.append(os.path.basename(p))
    return corners_list, names


def grade_count(n, rep):
    if n < MIN_IMAGES_HARD:
        rep.fail("count", f"only {n} usable images. Need at least {MIN_IMAGES_HARD}, "
                          f"ideally {MIN_IMAGES_OK}-30.")
    elif n < MIN_IMAGES_OK:
        rep.warn("count", f"{n} usable images; {MIN_IMAGES_OK}-30 is the standard "
                          f"range. Below 20 the distortion terms get noisy.")
    else:
        rep.ok("count", f"{n} usable images")


def grade_rms(rms, shape, rep):
    """RMS scales with resolution, so judge it at PROC_LONG_SIDE."""
    long_side = max(shape)
    equiv = rms * PROC_LONG_SIDE / float(long_side)
    note = "" if long_side == PROC_LONG_SIDE else \
        f" (= {equiv:.3f} px equivalent at {PROC_LONG_SIDE} px, which is what the "\
        f"plan's thresholds refer to)"
    if equiv > RMS_BAD:
        rep.fail("rms", f"RMS {rms:.3f} px at {long_side} px{note}. Over {RMS_BAD}. "
                        f"Almost always a board that is not perfectly flat, or "
                        f"blurred frames. Check the outlier list below first; if "
                        f"deleting those does not help, remount the board on "
                        f"stiffer card and reshoot.")
    elif equiv > RMS_GOOD:
        rep.warn("rms", f"RMS {rms:.3f} px at {long_side} px{note}. Usable but not "
                        f"clean; deleting one or two outlier frames often brings "
                        f"this under {RMS_GOOD}.")
    else:
        rep.ok("rms", f"RMS {rms:.3f} px at {long_side} px{note}")


def grade_fov(K, shape, rep):
    """Runs before everything else, because every later check trusts K.

    fx has a physical meaning: the horizontal field of view is
    2*atan(w / 2fx). A phone main camera is 60-80 degrees. If this comes out
    at 4 degrees or 150, K is not merely imprecise, it is wrong, and the tilt
    and outlier numbers below -- which come from solvePnP against this K --
    are meaningless. Stop and reshoot.
    """
    w, h = shape
    fovx = 2 * np.degrees(np.arctan(w / (2.0 * K[0, 0])))
    fovy = 2 * np.degrees(np.arctan(h / (2.0 * K[1, 1])))
    if not (FOV_MIN_DEG < fovx < FOV_MAX_DEG):
        rep.fail("field of view",
                 f"fx = {K[0,0]:.0f} px implies a {fovx:.1f}deg horizontal field of "
                 f"view on a {w}x{h} image. A phone is 60-80deg. This K is WRONG, "
                 f"not just imprecise — and note the RMS may still look excellent, "
                 f"because the solve traded focal length against distance and fitted "
                 f"the data perfectly with nonsense parameters. Reshoot with real "
                 f"tilt variety. Everything below this line is computed from this K "
                 f"and cannot be trusted.")
        return False
    rep.ok("field of view", f"{fovx:.1f}deg x {fovy:.1f}deg — plausible for a phone")
    return True


def grade_intrinsics(K, shape, rep):
    ar = K[0, 0] / K[1, 1]
    if not 0.97 < ar < 1.03:
        rep.warn("intrinsics", f"fx/fy = {ar:.3f}, unusual for a phone (expect ~1.0). "
                               f"Often a symptom of thin tilt diversity below.")
    else:
        rep.ok("intrinsics", f"fx/fy = {ar:.3f}, fx = {K[0,0]:.0f} px")
    ox = abs(K[0, 2] - shape[0] / 2) / shape[0]
    oy = abs(K[1, 2] - shape[1] / 2) / shape[1]
    if max(ox, oy) > 0.10:
        rep.warn("intrinsics",
                 f"principal point is {ox*100:.0f}%/{oy*100:.0f}% off centre. "
                 f"Principal-point estimation is ill-posed (the plan cites this from "
                 f"[1] §4.4); a large offset usually means poor frame coverage "
                 f"rather than a genuinely off-centre sensor.")


def grade_tilt(objp, corners_list, rep):
    """K-FREE tilt measurement, straight from the board homography.

    The earlier version derived board normals from solvePnP against K -- which
    is circular: if K is wrong, the poses are wrong, and the check that exists
    to catch a wrong K reports whatever the wrong K implies. Testing found
    exactly that, on a shoot whose fx was off by 17x.

    So measure the foreshortening directly. H maps board millimetres to pixels;
    its perspective row (h31, h32) is how fast the projective denominator
    changes across the board. Scaled by the board diagonal it is dimensionless:
    ~0 fronto-parallel, ~0.3-0.6 at a useful tilt. Its DIRECTION says which way
    the board leans. No K anywhere.

    Two things matter and measuring only the first is the trap:
      STRENGTH   is any board tilted at all?
      DIVERSITY  are they tilted in DIFFERENT directions? Twenty boards leaning
                 the same way constrain the same direction and are nearly as
                 degenerate as twenty flat ones.
    """
    L = float(np.linalg.norm(objp[:, :2].max(0) - objp[:, :2].min(0)))  # diagonal, mm
    vecs = []
    for c in corners_list:
        H, _ = cv2.findHomography(objp[:, :2].astype(np.float64),
                                  c.reshape(-1, 2).astype(np.float64), 0)
        if H is None or abs(H[2, 2]) < 1e-12:
            continue
        H = H / H[2, 2]
        vecs.append(np.array([H[2, 0], H[2, 1]]) * L)
    if not vecs:
        rep.warn("tilt", "could not fit board homographies; skipping the tilt check")
        return
    V = np.array(vecs)
    mag = np.linalg.norm(V, axis=1)
    strong = V[mag > TILT_MIN_STRENGTH]

    if len(strong) < 2:
        rep.fail("tilt",
                 f"board foreshortening is {mag.max():.2f} at most — the boards are "
                 f"nearly parallel with the sensor throughout. THIS IS THE CLASSIC "
                 f"TRAP: the solve reports a low RMS and hands you a focal length "
                 f"that is wrong, because nothing in the set distinguishes a large "
                 f"board far away from a small board close up. Reshoot with the "
                 f"board tilted 20-45deg.")
        return

    U = strong / np.linalg.norm(strong, axis=1, keepdims=True)
    G = np.clip(U @ U.T, -1, 1)
    iu = np.triu_indices(len(U), 1)
    diversity = float(np.median(np.degrees(np.arccos(G[iu]))))

    if diversity < TILT_MIN_DIVERSITY:
        rep.fail("tilt",
                 f"{len(strong)} boards are tilted, but the median angle between "
                 f"their tilt DIRECTIONS is only {diversity:.0f}deg — you leaned them "
                 f"all the same way. That is nearly as degenerate as not tilting at "
                 f"all, because every board constrains the same direction. Reshoot "
                 f"with the tilt in several DIFFERENT directions: left, right, "
                 f"top-away, bottom-away, plus a few rotated in-plane.")
    elif mag.max() > TILT_TOO_MUCH:
        rep.warn("tilt",
                 f"strong foreshortening on some frames ({mag.max():.2f}); beyond "
                 f"about {TILT_TOO_MUCH} corner localisation degrades and those "
                 f"frames may hurt more than they help — cross-check the outlier "
                 f"list.")
    else:
        rep.ok("tilt", f"{len(strong)}/{len(V)} boards meaningfully tilted "
                       f"(max foreshortening {mag.max():.2f}), median angle between "
                       f"tilt directions {diversity:.0f}deg")


def grade_scale(corners_list, rep):
    areas = []
    for c in corners_list:
        p = c.reshape(-1, 2)
        areas.append((p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
    ratio = max(areas) / max(1e-9, min(areas))
    if ratio < SCALE_MIN_RATIO:
        rep.warn("distance", f"largest board is only {ratio:.1f}x the area of the "
                             f"smallest — essentially one shooting distance. Vary it: "
                             f"some frames with the board filling three quarters of "
                             f"the frame, some filling a third.")
    else:
        rep.ok("distance", f"good distance variety (area ratio {ratio:.1f}x)")


def grade_coverage(corners_list, shape, rep, outdir):
    w, h = shape
    grid = np.zeros((COVERAGE_GRID, COVERAGE_GRID), int)
    for c in corners_list:
        p = c.reshape(-1, 2)
        gx = np.clip((p[:, 0] / w * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
        gy = np.clip((p[:, 1] / h * COVERAGE_GRID).astype(int), 0, COVERAGE_GRID - 1)
        for x, y in zip(gx, gy):
            grid[y, x] += 1

    filled = grid > 0
    frac = float(filled.mean())
    border = np.zeros_like(filled)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    edge_frac = float(filled[border].mean())

    lines = ["      frame coverage (each cell = 1/6 of the frame)"]
    for row in grid:
        lines.append("      " + " ".join(
            "." if v == 0 else ("o" if v < 20 else "#") for v in row))
    lines.append("      . never covered    o thin    # well covered")
    txt = "\n".join(lines)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "coverage.txt"), "w") as f:
        f.write(txt + "\n")

    if frac < COVERAGE_MIN or edge_frac < COVERAGE_EDGE_MIN:
        rep.fail("coverage",
                 f"corners reached only {frac*100:.0f}% of the frame and "
                 f"{edge_frac*100:.0f}% of the border cells. Distortion is estimated "
                 f"where you have data; empty edges mean those coefficients are "
                 f"extrapolated, and undistorting your scene photos will bend the "
                 f"edges the wrong way. Reshoot with the board pushed into every "
                 f"corner of the frame. Map in {outdir}/coverage.txt.")
    else:
        rep.ok("coverage", f"{frac*100:.0f}% of frame, {edge_frac*100:.0f}% of "
                           f"border cells covered")
    return txt


def grade_per_image(objp, corners_list, names, K, dist, rep):
    """Poses from solvePnP against the EXISTING K -- no recalibration."""
    errs, rvecs = [], []
    for c in corners_list:
        ok, rv, tv = cv2.solvePnP(objp, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            errs.append(np.nan); rvecs.append(np.zeros(3)); continue
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        # Flatten both to (N, 2) before comparing: the detectors return (N, 2)
        # while projectPoints returns (N, 1, 2), and cv2.norm rejects the
        # channel mismatch. Same defensive reshape as the graders above.
        a = c.reshape(-1, 2).astype(np.float64)
        b = proj.reshape(-1, 2).astype(np.float64)
        # RMS divides by sqrt(N), not N. Dividing by N (as the old OpenCV
        # tutorial does) understates the error by sqrt(N) -- ~7x for a 9x6
        # board -- and would make the RMS_GOOD floor below unreachable.
        errs.append(float(cv2.norm(a, b, cv2.NORM_L2) / np.sqrt(len(a))))
        rvecs.append(rv)
    errs = np.array(errs)
    med = float(np.nanmedian(errs)) if np.isfinite(errs).any() else float("nan")
    bad = [(names[i], errs[i]) for i in range(len(errs))
           if np.isfinite(errs[i]) and errs[i] > max(PER_IMAGE_ERR_MULT * med, RMS_GOOD)]
    if bad:
        detail = ", ".join(f"{n}({e:.2f}px)" for n, e in sorted(bad, key=lambda x: -x[1])[:8])
        rep.warn("outliers",
                 f"{len(bad)} images reproject far worse than the median ({med:.2f} px): "
                 f"{detail}. Usually motion blur, glare, or a board that flexed. Delete "
                 f"those files and re-run calibrate.py — the RMS should drop noticeably. "
                 f"If it does not, the board itself is not flat and deleting individual "
                 f"frames will not help.")
    else:
        rep.ok("outliers", f"no outlier images (median per-image RMS {med:.2f} px)")
    return rvecs


def undistort_check(path, K, dist, shape, outdir, rep):
    img = cv2.imread(path)
    if img is None:
        rep.warn("undistort", f"could not read {path}")
        return
    h, w = img.shape[:2]
    if (w, h) != tuple(shape):
        img = cv2.resize(img, tuple(shape), interpolation=cv2.INTER_AREA)
    und = cv2.undistort(img, K, dist, None, K)
    side = np.hstack([img, und])
    cv2.line(side, (shape[0], 0), (shape[0], shape[1]), (0, 0, 255), 2)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "undistort_check.jpg")
    cv2.imwrite(out, side, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rep.info("undistort",
             f"wrote {out} (original left, undistorted right). LOOK AT IT. Find a long "
             f"straight edge near the frame border — a door frame, a kerb, a railing — "
             f"and confirm it is straight on the right and visibly curved on the left. "
             f"This eyeball test is independent of the RMS and catches what the RMS "
             f"cannot.")


# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Grade a chessboard shoot against an existing calibration.")
    ap.add_argument("dirpath", help="the folder calibrate.py was run on")
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--calib", default=os.path.join(_root, "results", "calibration",
                                                    "calibration.json"))
    ap.add_argument("--format", default="jpg", dest="image_format")
    ap.add_argument("--detector", choices=DETECTORS, default=None,
                    help="corner detector; defaults to whatever calibration.json "
                         "records, so it always matches the calibration")
    ap.add_argument("--report-dir",
                    default=os.path.join(_root, "results", "calibration", "report"))
    ap.add_argument("--undistort-check", default=None,
                    help="a SCENE photo to undistort as a visual sanity check")
    a = ap.parse_args(argv)

    c = json.load(open(a.calib))
    K = np.array(c["K"], float).reshape(3, 3)
    dist = np.array(c["dist"], float).reshape(1, -1)
    shape = tuple(int(v) for v in c["resolution"])
    cw, ch = c.get("pattern_inner_corners", [9, 6])
    sq = float(c.get("square_size_mm", 25.0))
    rep = Report()

    print(hr(f"grade_calibration.py — {a.dirpath}"))
    print(f"  K from {a.calib}, solved at {shape[0]}x{shape[1]}, "
          f"pattern {cw}x{ch} inner corners")

    print(hr("re-detecting corners"))
    # Default to the detector the calibration actually used, so the two can
    # never drift apart again.
    detector = a.detector or c.get("detector", "classic")
    print(f"  detector: {detector}"
          + ("" if a.detector else "  (from calibration.json)"))
    corners_list, names = redetect(a.dirpath, (cw, ch), shape, a.image_format,
                                   rep, detector)
    print(f"  board found in {len(corners_list)} images")
    if len(corners_list) < 3:
        rep.fail("detection", "too few boards re-detected to grade anything")
        print(hr("verdict")); print(rep.render()); return 2

    objp = np.zeros((cw * ch, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cw, 0:ch].T.reshape(-1, 2) * sq

    print(hr("grading"))
    grade_count(len(corners_list), rep)
    k_ok = grade_fov(K, shape, rep)          # must run first; everything trusts K
    if "rms" in c:
        grade_rms(float(c["rms"]), shape, rep)
    grade_intrinsics(K, shape, rep)
    grade_tilt(objp, corners_list, rep)      # K-free, so still valid if k_ok is False
    grade_scale(corners_list, rep)
    if k_ok:
        grade_per_image(objp, corners_list, names, K, dist, rep)
    else:
        rep.info("outliers", "skipped — per-image errors come from solvePnP against "
                             "K, and K failed the field-of-view check above")
    print(grade_coverage(corners_list, shape, rep, a.report_dir))

    if a.undistort_check:
        undistort_check(a.undistort_check, K, dist, shape, a.report_dir, rep)

    print(hr("verdict"))
    print(rep.render())
    code = rep.code()
    print("\n  " + {0: "PASS — K is trustworthy, use it for all three scenes",
                    1: "WARN — usable, but read the warnings",
                    2: "FAIL — reshoot the board before shooting the scenes"}[code])
    print(f"  {rep.n_fail} failures, {rep.n_warn} warnings\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
