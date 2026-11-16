#!/usr/bin/env python3
"""
check_photos.py — capture quality assurance for the AI6121 SfM project.

WHAT THIS IS
    A pre-flight check you run on a folder of photos *before you leave campus*,
    to establish that the data can support a reconstruction at all. It answers
    one question: "if my pipeline fails on this scene, will it be my code's
    fault or my photos' fault?"

WHAT THIS IS NOT
    This is NOT part of the reconstruction pipeline. Nothing it computes is
    used by the SfM code. It therefore calls cv2.findFundamentalMat,
    findHomography, findEssentialMat and recoverPose freely, which the brief's
    prohibition does not touch because none of it enters the deliverable.
    Say exactly that in the report, in the same sentence that states the
    strict reading of the brief, so nobody mistakes it for an SfM shortcut.

USAGE
    The first argument is ANY folder of images -- relative or absolute. The
    scene type is inferred from the folder name (a name containing "easy",
    "medium" or "hard"), so --scene is only needed when the name does not say.

    # in the field, fast triage after each scene
    python check_photos.py photos/easy --fast

    # full check before you pack up; any path works
    python check_photos.py photos/easy_low
    python check_photos.py /Volumes/SD/shoot3 --scene medium

    # after Stage 1 calibration exists: adds the triangulation-angle report,
    # which is the only check that directly measures whether you walked far
    # enough between shots
    python check_photos.py photos/easy --calib calibration.json

    # back at the desk, exhaustive
    python check_photos.py photos/easy_combined --all-pairs --json report.json

EXIT CODES
    0  PASS      shoot is good
    1  WARN      usable, but read the warnings
    2  FAIL      reshoot something before you leave

INPUT
    A folder of JPEG or PNG files. Convert HEIC to JPEG first — OpenCV cannot
    read HEIC. Use a converter that PRESERVES EXIF; several phone-export paths
    strip it, and without EXIF the silent-ultrawide-switch check cannot run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from itertools import combinations

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONSTANTS — the single source of truth. The implementation plan quotes these
# numbers; the pipeline imports them from here rather than retyping them.
# Change one, change the plan. Every pixel threshold is stated at
# PROC_LONG_SIDE and is meaningless at any other resolution.
# ----------------------------------------------------------------------------

PROC_LONG_SIDE = 4032      # px. Native 3024x4032 portrait (plan, Part 1).
FAST_LONG_SIDE = 1600      # px. --fast only; expect slightly pessimistic counts.
EXPECT_WH = (3024, 4032)   # every scene image must be exactly this

BROKEN_INLIERS = 60        # a NEIGHBOURING pair below this is a broken link
WEAK_INLIERS = 150         # a NEIGHBOURING pair below this is thin
N_F_MIN = 30               # pipeline's hard floor for admitting an edge at all

EPS_HF = 0.75              # N_H / N_F at or above this => geometrically degenerate
LOWE_RATIO = 0.75          # matches the pipeline default
PIX_THRESH = 5.0           # px, RANSAC inlier threshold for F and H at PROC_LONG_SIDE.
                           # Scaled at the call site when --fast lowers the resolution,
                           # so a fast run stays commensurable with a full one.

NFEATURES_FIELD = 4000     # the pipeline uses 8000; expect this to be pessimistic
NFEATURES_FAST = 1500

MIN_FEATURES = 800         # an image below this is too smooth / too blurred
BLUR_ABS_FLOOR = 40.0      # variance of Laplacian; below this is definitely soft
BLUR_REL_FLOOR = 0.40      # ...or below this fraction of the folder median
CLIP_FRAC_MAX = 0.08       # >8% of pixels crushed to 0 or blown to 255

# Triangulation angle expectations for NEIGHBOURING pairs (needs --calib).
ANGLE_TOO_SMALL = 2.0      # deg. Below this you rotated instead of translating.
ANGLE_TOO_LARGE = 15.0     # deg. Above this the overlap is probably too thin.

MIN_IMAGES = 30            # the plan asks for 40-60 per scene
RANDOM_LONG_PAIRS = 60     # long-range pairs sampled in full (non --all-pairs) mode

SEED = 6121

_TTY = sys.stdout.isatty()


def progress(msg):
    """Only emit carriage-return progress on a real terminal, so piping the
    output to a file or a log does not produce thousands of lines."""
    if _TTY:
        print(f"\r{msg}", end="", flush=True)


# ----------------------------------------------------------------------------
# small reporting helpers
# ----------------------------------------------------------------------------

class Report:
    """Collects findings and decides the exit code."""

    def __init__(self):
        self.rows = []          # (level, section, message)
        self.data = {}          # machine-readable payload for --json

    def ok(self, section, msg):
        self.rows.append(("OK", section, msg))

    def warn(self, section, msg):
        self.rows.append(("WARN", section, msg))

    def fail(self, section, msg):
        self.rows.append(("FAIL", section, msg))

    def info(self, section, msg):
        self.rows.append(("INFO", section, msg))

    @property
    def n_fail(self):
        return sum(1 for lv, _, _ in self.rows if lv == "FAIL")

    @property
    def n_warn(self):
        return sum(1 for lv, _, _ in self.rows if lv == "WARN")

    def exit_code(self):
        if self.n_fail:
            return 2
        return 1 if self.n_warn else 0

    def render(self):
        order = {"FAIL": 0, "WARN": 1, "OK": 2, "INFO": 3}
        width = max((len(s) for _, s, _ in self.rows), default=10)
        lines = []
        for lv in ("FAIL", "WARN", "OK", "INFO"):
            chunk = [r for r in self.rows if r[0] == lv]
            for _, section, msg in chunk:
                lines.append(f"  [{lv:4s}] {section:<{width}s}  {msg}")
        del order
        return "\n".join(lines)


def hr(title=""):
    bar = "-" * 78
    return f"\n{bar}\n{title}\n{bar}" if title else bar


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(name):
    """Sort img2 before img10. Plain alphabetical order puts sfm10 between sfm1
    and sfm2, which silently makes 'neighbouring pairs' mean 'photos from
    opposite sides of the object' and reports 0 inliers on perfectly good data.
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def list_images(folder):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder), key=natural_key)
             if os.path.splitext(f)[1] in IMG_EXT]
    heic = [f for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in (".heic", ".heif")]
    return files, heic


def read_exif(path):
    """Best-effort EXIF read. Returns {} rather than raising."""
    out = {}
    try:
        from PIL import Image, ExifTags
    except Exception:
        return {"_error": "Pillow not installed"}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return out
            base = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
            for k in ("Make", "Model", "Orientation"):
                if k in base:
                    out[k] = base[k]
            try:
                sub = ex.get_ifd(0x8769)          # Exif IFD
            except Exception:
                sub = {}
            subn = {ExifTags.TAGS.get(k, k): v for k, v in sub.items()}
            for k in ("LensModel", "LensMake", "FocalLength",
                      "FocalLengthIn35mmFilm", "DigitalZoomRatio",
                      "ExposureTime", "ISOSpeedRatings",
                      "PhotographicSensitivity", "ExposureMode",
                      "WhiteBalance", "SceneCaptureType"):
                if k in subn:
                    v = subn[k]
                    try:
                        v = float(v) if hasattr(v, "numerator") else v
                    except Exception:
                        pass
                    out[k] = v
    except Exception as e:                       # noqa: BLE001
        out["_error"] = repr(e)
    return out


def load_resized(path, long_side):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    scale = long_side / float(max(h, w))
    if scale < 1.0:
        img_s = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        img_s = img
        scale = 1.0
    return img_s, (w, h)


# ----------------------------------------------------------------------------
# per-image checks
# ----------------------------------------------------------------------------

def per_image_pass(paths, long_side, nfeatures, rep):
    """Loads every image once: dimensions, EXIF, blur, exposure, features."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    recs = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img, orig_wh = load_resized(p, long_side)
        if img is None:
            rep.fail("load", f"{os.path.basename(p)} could not be decoded")
            recs.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        clip = float(((gray <= 2) | (gray >= 253)).mean())
        kp, desc = sift.detectAndCompute(gray, None)
        recs.append({
            "path": p,
            "name": os.path.basename(p),
            "orig_wh": orig_wh,
            "proc_wh": (img.shape[1], img.shape[0]),
            "blur": blur,
            "clip": clip,
            "n_kp": 0 if desc is None else int(desc.shape[0]),
            "kp": np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32),
            "desc": desc,
            "exif": read_exif(p),
        })
        progress(f"  scanning {i + 1}/{len(paths)} ...")
    print(f"\r  scanned {len(paths)} images in {time.time() - t0:.1f}s" + " " * 12)
    return [r for r in recs if r is not None]


def check_dimensions(recs, rep):
    sizes = {}
    for r in recs:
        sizes.setdefault(r["orig_wh"], []).append(r["name"])
    rep.data["dimensions"] = {f"{w}x{h}": len(v) for (w, h), v in sizes.items()}
    if len(sizes) == 1:
        (w, h) = next(iter(sizes))
        if (w, h) != EXPECT_WH:
            rep.fail("dimensions",
                     f"all images are {w}x{h}, but the plan pins {EXPECT_WH[0]}x"
                     f"{EXPECT_WH[1]} portrait. If you shot landscape, every pixel "
                     f"threshold and the calibrated K are for the other orientation. "
                     f"Reshoot in portrait, or change EXPECT_WH and rescale every "
                     f"threshold — do not silently proceed.")
        else:
            rep.ok("dimensions", f"all {len(recs)} images are {w}x{h} portrait")
    else:
        biggest = max(sizes.items(), key=lambda kv: len(kv[1]))
        others = {k: v for k, v in sizes.items() if k != biggest[0]}
        detail = "; ".join(f"{w}x{h}: {', '.join(v[:4])}{' ...' if len(v) > 4 else ''}"
                           for (w, h), v in others.items())
        rep.fail("dimensions",
                 f"{len(sizes)} distinct image sizes in one folder. One K cannot "
                 f"describe them. Majority is {biggest[0][0]}x{biggest[0][1]}; "
                 f"odd ones out -> {detail}. If some are transposed, that is the "
                 f"EXIF orientation flag: re-export with orientation applied and "
                 f"stripped, or drop those frames.")


def check_exif(recs, rep):
    have = [r for r in recs if r["exif"] and "_error" not in r["exif"]]
    if not have:
        rep.warn("exif",
                 "no readable EXIF on any image. The silent-ultrawide-lens check "
                 "cannot run. If you converted from HEIC, re-convert with a tool "
                 "that preserves metadata (e.g. `magick mogrify -format jpg`). "
                 "Not fatal, but you lose the one check that catches a lens "
                 "switch no inlier count reveals.")
        rep.data["exif"] = None
        return

    if len(have) < len(recs):
        rep.warn("exif", f"{len(recs) - len(have)} of {len(recs)} images have no "
                         f"readable EXIF; checked the remaining {len(have)}")

    keys = ["Model", "LensModel", "FocalLength", "FocalLengthIn35mmFilm",
            "DigitalZoomRatio"]
    summary = {}
    for k in keys:
        groups = {}
        for r in have:
            v = r["exif"].get(k, "<missing>")
            groups.setdefault(str(v), []).append(r["name"])
        summary[k] = {kk: len(vv) for kk, vv in groups.items()}
        if len(groups) <= 1:
            continue
        majority = max(groups.items(), key=lambda kv: len(kv[1]))
        odd = {kk: vv for kk, vv in groups.items() if kk != majority[0]}
        detail = "; ".join(
            f"{kk} -> {', '.join(vv[:5])}{' ...' if len(vv) > 5 else ''}"
            for kk, vv in odd.items())
        level = rep.fail if k in ("LensModel", "FocalLength",
                                  "FocalLengthIn35mmFilm") else rep.warn
        level("exif",
              f"{k} is not constant across the folder (majority '{majority[0]}' "
              f"on {len(majority[1])} images). Offenders: {detail}. A changed "
              f"focal length or lens means our single K is wrong for those "
              f"frames — remove them or reshoot the scene with the lens locked.")

    zooms = [float(r["exif"]["DigitalZoomRatio"]) for r in have
             if isinstance(r["exif"].get("DigitalZoomRatio"), (int, float))]
    if zooms and max(zooms) > 1.0001:
        rep.fail("exif", f"digital zoom used on some frames (max ratio "
                         f"{max(zooms):.2f}). Reshoot those without zooming.")

    orients = {int(r["exif"]["Orientation"]) for r in have
               if isinstance(r["exif"].get("Orientation"), int)}
    if orients and orients != {1}:
        rep.warn("exif",
                 f"EXIF Orientation values {sorted(orients)} present (1 = normal). "
                 f"cv2.imread applies these, so images may arrive transposed. "
                 f"The dimensions check above is the one that matters; if it "
                 f"passed, you are fine.")

    rep.data["exif"] = summary
    if rep.n_fail == 0:
        m = have[0]["exif"].get("Model", "?")
        lens = have[0]["exif"].get("LensModel", "?")
        rep.ok("exif", f"consistent across {len(have)} images (Model={m}, Lens={lens})")


def check_quality(recs, rep):
    blurs = np.array([r["blur"] for r in recs])
    med = float(np.median(blurs))
    soft = [r for r in recs
            if r["blur"] < max(BLUR_ABS_FLOOR, BLUR_REL_FLOOR * med)]
    rep.data["blur_median"] = med
    if soft:
        names = ", ".join(f"{r['name']}({r['blur']:.0f})" for r in soft[:8])
        (rep.fail if len(soft) > len(recs) * 0.15 else rep.warn)(
            "blur",
            f"{len(soft)}/{len(recs)} images are soft relative to the folder "
            f"median ({med:.0f}): {names}{' ...' if len(soft) > 8 else ''}. "
            f"Delete or reshoot them — a blurred frame contributes bad features "
            f"and a bad pose, and costs more than the coverage it adds.")
    else:
        rep.ok("blur", f"all images sharp (median Laplacian variance {med:.0f})")

    clipped = [r for r in recs if r["clip"] > CLIP_FRAC_MAX]
    if clipped:
        names = ", ".join(f"{r['name']}({r['clip'] * 100:.0f}%)" for r in clipped[:8])
        rep.warn("exposure",
                 f"{len(clipped)} images have >{CLIP_FRAC_MAX * 100:.0f}% crushed "
                 f"or blown pixels: {names}. Blown highlights carry no features. "
                 f"Overcast light or open shade, as the plan says.")
    else:
        rep.ok("exposure", "no significant clipping")

    thin = [r for r in recs if r["n_kp"] < MIN_FEATURES]
    counts = [r["n_kp"] for r in recs]
    rep.data["features_median"] = int(np.median(counts))
    if thin:
        names = ", ".join(f"{r['name']}({r['n_kp']})" for r in thin[:8])
        (rep.fail if len(thin) > len(recs) * 0.25 else rep.warn)(
            "features",
            f"{len(thin)}/{len(recs)} images yield under {MIN_FEATURES} features: "
            f"{names}. The surface is too smooth or the frame is mostly sky. "
            f"Note the pipeline uses nfeatures=8000 where this tool uses "
            f"{NFEATURES_FIELD}, so this is pessimistic — but not by much.")
    else:
        rep.ok("features",
               f"median {int(np.median(counts))} features per image "
               f"(min {min(counts)}, max {max(counts)})")


# ----------------------------------------------------------------------------
# pair geometry
# ----------------------------------------------------------------------------

def _flann():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def match_pair(ra, rb, matcher):
    """Lowe ratio test + mutual (cross) check. Returns matched point arrays."""
    da, db = ra["desc"], rb["desc"]
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    def ratio(d1, d2):
        knn = matcher.knnMatch(d1, d2, k=2)
        best = {}
        for m in knn:
            if len(m) < 2:
                continue
            if m[0].distance < LOWE_RATIO * m[1].distance:
                best[m[0].queryIdx] = m[0].trainIdx
        return best

    ab = ratio(da, db)
    ba = ratio(db, da)
    pairs = [(i, j) for i, j in ab.items() if ba.get(j, -1) == i]   # mutual check
    if not pairs:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    return ra["kp"][ia], rb["kp"][ib]


_THRESH = PIX_THRESH       # set once in main() to match the working resolution


def _fit_F(p1, p2):
    if len(p1) < 8:
        return 0
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC,
                                         _THRESH, 0.999, 10000)
    except cv2.error:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC,
                                         _THRESH, 0.999)
    if F is None or mask is None:
        return 0
    return int(mask.sum())


def _fit_H(p1, p2):
    if len(p1) < 4:
        return 0
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, _THRESH,
                                 maxIters=10000, confidence=0.999)
    if H is None or mask is None:
        return 0
    return int(mask.sum())


def median_tri_angle(p1, p2, K):
    """Median triangulation angle in degrees for a calibrated pair.

    Uses OpenCV end to end on purpose: this is a data check, not the pipeline.
    """
    if len(p1) < 8:
        return None, 0
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=_THRESH)
    if E is None or mask is None or E.shape != (3, 3):
        return None, 0
    inl = mask.ravel().astype(bool)
    if inl.sum() < 8:
        return None, int(inl.sum())
    n_good, R, t, _ = cv2.recoverPose(E, p1[inl], p2[inl], K)
    if n_good < 5:
        return None, int(inl.sum())

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, p1[inl].T, p2[inl].T)
    w = X[3]
    ok = np.abs(w) > 1e-9
    X = (X[:3, ok] / w[ok]).T                      # (N,3) in camera-1 frame
    if len(X) < 5:
        return None, int(inl.sum())

    C1 = np.zeros(3)
    C2 = (-R.T @ t).ravel()
    v1 = X - C1
    v2 = X - C2
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    good = (n1 > 1e-9) & (n2 > 1e-9)
    if good.sum() < 5:
        return None, int(inl.sum())
    cosang = np.clip((v1[good] * v2[good]).sum(1) / (n1[good] * n2[good]), -1, 1)
    return float(np.degrees(np.arccos(cosang)).mean().item()
                 if False else np.median(np.degrees(np.arccos(cosang)))), int(inl.sum())


def build_pair_list(n, fast, all_pairs):
    if all_pairs:
        return list(combinations(range(n), 2)), set(), set()
    neigh = [(i, i + 1) for i in range(n - 1)]
    if fast:
        return neigh, set(neigh), set()
    wide = [(i, i + k) for k in (2, 3) for i in range(n - k)]
    loop = [(0, n - 1), (0, n - 2), (1, n - 1)] if n >= 4 else []
    rng = random.Random(SEED)
    pool = [p for p in combinations(range(n), 2) if p[1] - p[0] > 3]
    rng.shuffle(pool)
    longr = pool[:RANDOM_LONG_PAIRS]
    seen, out = set(), []
    for p in neigh + wide + loop + longr:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out, set(neigh), set(loop)


def pair_pass(recs, pairs, K, rep):
    matcher = _flann()
    results = {}
    t0 = time.time()
    for n, (i, j) in enumerate(pairs):
        p1, p2 = match_pair(recs[i], recs[j], matcher)
        nf = _fit_F(p1, p2)
        nh = _fit_H(p1, p2) if nf >= 8 else 0
        ang, ne = (None, 0)
        if K is not None and nf >= N_F_MIN:
            ang, ne = median_tri_angle(p1, p2, K)
        results[(i, j)] = {"raw": int(len(p1)), "n_f": nf, "n_h": nh,
                           "n_e": ne, "hf": (nh / nf) if nf else 0.0,
                           "angle": ang}
        progress(f"  pairs {n + 1}/{len(pairs)} ...")
    print(f"\r  matched {len(pairs)} pairs in {time.time() - t0:.1f}s" + " " * 12)
    rep.data["n_pairs_tested"] = len(pairs)
    return results


def grade_pairs(recs, results, neigh, loop, scene, rep):
    names = [r["name"] for r in recs]

    broken = [(p, v) for p, v in results.items()
              if p in neigh and v["n_f"] < BROKEN_INLIERS]
    weak = [(p, v) for p, v in results.items()
            if p in neigh and BROKEN_INLIERS <= v["n_f"] < WEAK_INLIERS]

    if broken:
        detail = ", ".join(f"{names[i]}~{names[j]}({v['n_f']})"
                           for (i, j), v in broken[:8])
        rep.fail("links",
                 f"{len(broken)} NEIGHBOURING pairs under {BROKEN_INLIERS} inliers: "
                 f"{detail}{' ...' if len(broken) > 8 else ''}. The chain is broken "
                 f"there — the reconstruction will split into components. Reshoot "
                 f"the gap with intermediate viewpoints.")
    elif weak:
        detail = ", ".join(f"{names[i]}~{names[j]}({v['n_f']})"
                           for (i, j), v in weak[:8])
        rep.warn("links",
                 f"{len(weak)} neighbouring pairs are thin (under {WEAK_INLIERS}): "
                 f"{detail}. Usable, but a few extra frames in those gaps is cheap "
                 f"insurance.")
    else:
        vals = [v["n_f"] for p, v in results.items() if p in neigh]
        if vals:
            rep.ok("links", f"all {len(vals)} neighbouring pairs healthy "
                            f"(median {int(np.median(vals))} inliers, min {min(vals)})")

    # --- degeneracy -----------------------------------------------------
    tested = [(p, v) for p, v in results.items() if v["n_f"] >= N_F_MIN]
    deg = [(p, v) for p, v in tested if v["hf"] >= EPS_HF]
    frac = len(deg) / max(1, len(tested))
    rep.data["degenerate_fraction"] = frac
    if tested:
        if frac > 0.95:
            rep.fail("degeneracy",
                     f"{frac * 100:.0f}% of pairs are geometrically degenerate "
                     f"(N_H/N_F >= {EPS_HF}). Either you never translated, or the "
                     f"scene is entirely planar. No processing recovers depth from "
                     f"this — step sideways between shots, and include some relief.")
        elif scene == "easy" and frac > 0.30:
            deg_neigh = [(p, v) for p, v in deg if p in neigh]
            detail = ", ".join(f"{names[i]}~{names[j]}({v['hf']:.2f})"
                               for (i, j), v in deg_neigh[:8])
            rep.fail("degeneracy",
                     f"{frac * 100:.0f}% of pairs degenerate on an EASY scene, which "
                     f"should be an isolated object with depth. Reads as 'you rotated "
                     f"instead of stepping sideways'. Offending neighbours: {detail}")
        elif frac > 0.30:
            rep.warn("degeneracy",
                     f"{frac * 100:.0f}% of pairs degenerate (N_H/N_F >= {EPS_HF}). "
                     f"On a {scene} scene a facade or flat wall trips this "
                     f"legitimately, so this is a warning, not an error. But note "
                     f"the consequence: those pairs are labelled `planar` and cannot "
                     f"seed the reconstruction (plan, Stage 5c). If ALL of them are, "
                     f"you will be exercising the homography seed fallback in 5d.")
        else:
            rep.ok("degeneracy", f"{frac * 100:.0f}% of pairs degenerate — normal")

    # --- loop closure ---------------------------------------------------
    if loop:
        best = max((results[p]["n_f"] for p in loop if p in results), default=0)
        if scene == "easy":
            if best < BROKEN_INLIERS:
                rep.warn("loop closure",
                         f"first and last frames share only {best} inliers. If you "
                         f"walked a full circle they should reconnect; without that "
                         f"the loop never closes and drift accumulates with nothing "
                         f"to correct it. Shoot a few more frames past the start.")
            else:
                rep.ok("loop closure", f"first/last frames reconnect ({best} inliers)")
        else:
            rep.info("loop closure",
                     f"first/last frames share {best} inliers (not graded on a "
                     f"{scene} scene — the path is not a circle)")

    # --- connectivity ---------------------------------------------------
    parent = list(range(len(recs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j), v in results.items():
        if v["n_f"] >= N_F_MIN:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    comps = {}
    for i in range(len(recs)):
        comps.setdefault(find(i), []).append(names[i])
    rep.data["components"] = [sorted(v) for v in comps.values()]
    if len(comps) == 1:
        rep.ok("connectivity", f"all {len(recs)} images in one component")
    else:
        sizes = sorted((len(v) for v in comps.values()), reverse=True)
        singles = [v[0] for v in comps.values() if len(v) == 1]
        rep.fail("connectivity",
                 f"{len(comps)} disconnected components over the tested pairs "
                 f"(sizes {sizes}). The reconstruction cannot merge these. "
                 f"Isolated frames: {', '.join(singles[:10])}"
                 f"{' ...' if len(singles) > 10 else ''}. Note this is judged on the "
                 f"tested subgraph — re-run with --all-pairs before believing it.")

    # --- triangulation angles (requires --calib) -------------------------
    angs = [v["angle"] for p, v in results.items()
            if p in neigh and v["angle"] is not None]
    if angs:
        med = float(np.median(angs))
        rep.data["neighbour_angle_median_deg"] = med
        small = [p for p in neigh
                 if results.get(p, {}).get("angle") is not None
                 and results[p]["angle"] < ANGLE_TOO_SMALL]
        if med < ANGLE_TOO_SMALL:
            rep.fail("baseline",
                     f"median neighbouring triangulation angle is {med:.1f}deg, under "
                     f"{ANGLE_TOO_SMALL}deg. You barely translated between shots. "
                     f"Depth is not recoverable at this baseline no matter how good "
                     f"the code is.")
        elif med > ANGLE_TOO_LARGE:
            rep.warn("baseline",
                     f"median neighbouring triangulation angle is {med:.1f}deg — wide. "
                     f"Good for depth, risky for matching. Check the link counts "
                     f"above; if they are thin, you stepped too far.")
        else:
            rep.ok("baseline",
                   f"median neighbouring triangulation angle {med:.1f}deg "
                   f"(target 6-9deg for the easy scene's full loop)")
        if small and med >= ANGLE_TOO_SMALL:
            rep.warn("baseline",
                     f"{len(small)} individual neighbouring pairs under "
                     f"{ANGLE_TOO_SMALL}deg: "
                     f"{', '.join(f'{names[i]}~{names[j]}' for i, j in small[:8])}. "
                     f"Those specific pairs will be labelled panoramic and excluded "
                     f"from triangulation (plan, Stage 3b/6).")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def load_calib(path, rep, proc_wh):
    with open(path) as f:
        c = json.load(f)
    K = np.array(c["K"], dtype=np.float64).reshape(3, 3)
    calib_wh = tuple(c.get("resolution") or c.get("calib_resolution") or (0, 0))
    if calib_wh and tuple(calib_wh) != tuple(proc_wh):
        sx = proc_wh[0] / float(calib_wh[0])
        sy = proc_wh[1] / float(calib_wh[1])
        if abs(sx - sy) > 1e-3:
            rep.fail("calibration",
                     f"calibration resolution {calib_wh} and processing resolution "
                     f"{proc_wh} have different aspect ratios. K cannot be rescaled "
                     f"between them. Recalibrate at the processing resolution.")
            return None
        K = K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
        rep.info("calibration",
                 f"rescaled K from {calib_wh} to {proc_wh} (factor {sx:.4f}) for this "
                 f"check only. The pipeline must use ONE K at ONE resolution — see "
                 f"the plan, Part 1 and Stage 1.")
    else:
        rep.ok("calibration", f"K loaded at {proc_wh}")
    if "rms" in c:
        rms = float(c["rms"])
        (rep.warn if rms > 1.0 else rep.ok)(
            "calibration", f"reported calibration RMS {rms:.3f} px")
    return K


SCENE_TYPES = ("easy", "medium", "hard")


SCENE_FILE = ".scene"


def infer_scene(folder):
    """Work out the scene type for a free-form folder path.

    Precedence: a .scene file inside the folder, then the folder name, then
    'easy'. The .scene file is the reliable one, because folders get named
    after their subject ("yunnan_rock") far more often than after a grading
    category, and it survives renaming and moving the folder. Write it once:

        echo easy > photos/yunnan_rock/.scene

    Returns (scene, why). The fallback is 'easy' -- the old unconditional
    default -- and it says so out loud, because silently grading a facade as
    an orbit is the mistake this is meant to prevent.
    """
    sf = os.path.join(folder, SCENE_FILE)
    if os.path.isfile(sf):
        try:
            with open(sf) as f:
                val = f.read().strip().lower()
            if val in SCENE_TYPES:
                return val, f"from {SCENE_FILE} in the folder"
            print(f"  WARNING: {sf} says '{val}', not one of "
                  f"{'/'.join(SCENE_TYPES)} -- ignoring it")
        except OSError as e:
            print(f"  WARNING: could not read {sf}: {e!r}")

    name = os.path.basename(os.path.normpath(folder)).lower()
    for s in SCENE_TYPES:
        if s in name:
            return s, f"inferred from folder name '{name}'"
    return "easy", (f"no {SCENE_FILE} file and folder name '{name}' does not "
                    f"say; assuming easy -- pass --scene, or write one with "
                    f"`echo <type> > {os.path.join(folder, SCENE_FILE)}`")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Capture QA for the AI6121 SfM project. Not part of the pipeline.")
    ap.add_argument("folder", help="folder of JPEG/PNG images for ONE scene")
    ap.add_argument("--scene", choices=SCENE_TYPES, default=None,
                    help="changes how degeneracy and loop closure are graded. "
                         "Default: inferred from the folder name.")
    ap.add_argument("--fast", action="store_true",
                    help="field triage: lower resolution, fewer features, "
                         "consecutive pairs only")
    ap.add_argument("--all-pairs", action="store_true",
                    help="exhaustive pair testing (slow; do this at your desk)")
    ap.add_argument("--calib", default=None,
                    help="calibration JSON with a 'K' (and ideally 'resolution'); "
                         "enables the triangulation-angle report")
    ap.add_argument("--json", default=None, help="write a machine-readable report here")
    ap.add_argument("--long-side", type=int, default=None,
                    help=f"override processing long side (default {PROC_LONG_SIDE}, "
                         f"{FAST_LONG_SIDE} with --fast)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.folder):
        print(f"not a folder: {args.folder}")
        return 2
    if args.scene:
        scene, why = args.scene, "given on the command line"
    else:
        scene, why = infer_scene(args.folder)
    args.scene = scene

    rep = Report()
    long_side = args.long_side or (FAST_LONG_SIDE if args.fast else PROC_LONG_SIDE)
    global _THRESH
    _THRESH = PIX_THRESH * long_side / float(PROC_LONG_SIDE)   # keep px thresholds
                                                               # commensurable at any
                                                               # working resolution
    nfeat = NFEATURES_FAST if args.fast else NFEATURES_FIELD

    print(hr(f"check_photos.py — {args.folder}  [scene={args.scene}"
             f"{', fast' if args.fast else ''}]"))
    print(f"  scene type '{scene}' -- {why}")
    print(f"  processing at {long_side} px long side, nfeatures={nfeat}")
    if args.fast:
        print("  NOTE: --fast is triage only. Run the full check before you pack up.")

    paths, heic = list_images(args.folder)
    if heic:
        rep.fail("input", f"{len(heic)} HEIC/HEIF files present — OpenCV cannot read "
                          f"them. Convert to JPEG with a converter that preserves "
                          f"EXIF, then re-run.")
    if not paths:
        rep.fail("input", "no JPEG/PNG images found")
        print(rep.render())
        return 2
    plain = sorted(os.path.basename(p) for p in paths)
    if plain != [os.path.basename(p) for p in paths]:
        rep.warn("filenames",
                 "filenames are not zero-padded, so plain alphabetical order "
                 "differs from shooting order (sfm10 sorts between sfm1 and "
                 "sfm2). This tool sorts naturally and is fine, but your "
                 "pipeline, COLMAP and your file browser will all disagree. "
                 "Rename once now: "
                 "`n=1; for f in $(ls *.jpg | sort -V); do "
                 "mv \"$f\" \"$(printf 'img_%03d.jpg' $n)\"; n=$((n+1)); done`")

    if len(paths) < MIN_IMAGES:
        rep.warn("input", f"only {len(paths)} images; the plan asks for 40-60 per "
                          f"scene, and ablation 6 needs 50 to subsample from")
    else:
        rep.ok("input", f"{len(paths)} images")

    print(hr("per-image checks"))
    recs = per_image_pass(paths, long_side, nfeat, rep)
    if len(recs) < 2:
        print(rep.render())
        return 2
    check_dimensions(recs, rep)
    check_exif(recs, rep)
    check_quality(recs, rep)

    K = None
    if args.calib:
        print(hr("calibration"))
        try:
            K = load_calib(args.calib, rep, recs[0]["proc_wh"])
        except Exception as e:                                   # noqa: BLE001
            rep.warn("calibration", f"could not load {args.calib}: {e!r}. "
                                    f"Skipping the triangulation-angle report.")

    print(hr("pairwise geometry"))
    pairs, neigh, loop = build_pair_list(len(recs), args.fast, args.all_pairs)
    print(f"  testing {len(pairs)} pairs "
          f"({'exhaustive' if args.all_pairs else 'neighbourhood + loop + sample'})")
    results = pair_pass(recs, pairs, K, rep)
    grade_pairs(recs, results, neigh, loop, args.scene, rep)

    print(hr("verdict"))
    print(rep.render())
    code = rep.exit_code()
    verdict = {0: "PASS — data is good, any later failure is your code's fault",
               1: "WARN — usable, but read the warnings above",
               2: "FAIL — fix something before you leave campus"}[code]
    print(f"\n  {verdict}")
    print(f"  {rep.n_fail} failures, {rep.n_warn} warnings\n")

    if args.json:
        rep.data["verdict"] = verdict
        rep.data["scene"] = args.scene
        rep.data["long_side"] = long_side
        rep.data["findings"] = [{"level": lv, "section": s, "message": m}
                                for lv, s, m in rep.rows]
        rep.data["pairs"] = {f"{i}-{j}": v for (i, j), v in results.items()}
        rep.data["images"] = [{k: v for k, v in r.items()
                               if k not in ("kp", "desc")} for r in recs]
        with open(args.json, "w") as f:
            json.dump(rep.data, f, indent=2, default=str)
        print(f"  wrote {args.json}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
