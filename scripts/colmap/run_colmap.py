#!/usr/bin/env python3
"""
run_colmap.py - Run a full COLMAP sparse reconstruction on a photo folder and
print the benchmark numbers.

This is the heavyweight counterpart to check_photos.py. check_photos.py answers
"are neighbouring photos linked?" in seconds per pair. This answers "does the
whole set actually reconstruct?" - which is the number you quote in a
benchmark - and it costs minutes, not seconds.

It reports, for the largest reconstructed model:
    registered images / total        the headline number
    sparse 3D points
    mean track length                observations per 3D point
    mean observations per image
    mean reprojection error (px)     computed the same way COLMAP's
                                     model_analyzer does (observation-weighted)
    wall-clock time per stage        extraction / matching / mapping

...plus the two-view geometry stats straight out of COLMAP's own database, so
you can put COLMAP's RANSAC inlier counts next to the ones check_photos.py
printed for the same folder.

Usage:
    python run_colmap.py /path/to/photos
    python run_colmap.py /path/to/photos --max-image-size 1600   # much faster
    python run_colmap.py /path/to/photos --matcher sequential    # long walks
    python run_colmap.py /path/to/photos --workspace out/hive --force
    python run_colmap.py /path/to/photos --reuse        # re-report, no recompute
    python run_colmap.py /path/to/photos --no-yup       # skip the viewing copy

Output, in results/colmap_out/<folder name>/:
    benchmark.json, database.db
    sparse/0_raw/               COLMAP's own gauge (+ txt/). Quote numbers from here.
    sparse/0_yup/               same model rotated so gravity = +Y. Viewing only.
    <name>_yup2.ply             points, yup gauge
    <name>_with_cameras.ply     points + one red vertex per camera, yup gauge
    preview.jpg                 top-down + side view
The yup step (make_yup.py) runs only when COLMAP built exactly one model; with
--no-yup, or on a split result, the model stays in sparse/0 as COLMAP wrote it.

Feature extraction dominates the wall clock and runs on the CPU unless your
COLMAP was built with CUDA, so --max-image-size is the knob that matters if
you are waiting too long. For the same reason, only time a run that has the
machine to itself - a second job halved the effective core count here and
turned 3m30s into 17m for identical inputs.

Two caveats for anything you publish from these numbers:
  * COLMAP's mapper is not deterministic. Repeat runs on the same photos move
    the point count and reprojection error by a fraction of a percent, so
    quote a range from a few runs rather than a single figure.
  * Registration percentage alone is not enough. Check `models reconstructed`
    too: COLMAP happily returns several disconnected models, and a headline
    "90% registered" can mean two halves that never joined.

Backends (auto-detected, in this order):
    1. the `colmap` command line binary      brew install colmap
    2. the pycolmap python module            pip install pycolmap
"""

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Tuning. Kept deliberately close to check_photos.py so the two scripts are
# describing the same photographs rather than two different feature budgets.
# ---------------------------------------------------------------------------
N_FEATURES = 4000        # SIFT keypoints per image, matching check_photos.py.
                         # COLMAP then emits up to max_num_orientations (2)
                         # descriptors per keypoint, so the per-image count in
                         # the database lands above this - roughly 1.5x.
CAMERA_MODEL = "SIMPLE_RADIAL"   # sane default for a phone / compact camera
MAX_IMAGE_SIZE = 3200    # COLMAP's own default; --max-image-size overrides it.
                         # check_photos.py works at 1600, so pass that if you
                         # want the two scripts on identical pixels.
SEQ_OVERLAP = 10         # sequential matcher: how many neighbours ahead to try

# Verdict thresholds, in fraction of images registered into one model.
GOOD_REGISTERED = 0.95
WEAK_REGISTERED = 0.80

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

MAX_PAIR_ID = 2147483647


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def natural_key(name):
    """Sort sfm2.jpg before sfm10.jpg, not after."""
    parts, digits = [], ""
    for ch in name:
        if ch.isdigit():
            digits += ch
        else:
            if digits:
                parts.append(int(digits))
                digits = ""
            parts.append(ch.lower())
    if digits:
        parts.append(int(digits))
    return parts


def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(IMAGE_EXTS)],
                  key=natural_key)


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:04.1f}s"


class Stopwatch:
    """Times named stages and keeps them in insertion order."""

    def __init__(self):
        self.stages = {}

    def run(self, name, fn):
        print(f"  [{name}] running...", flush=True)
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        self.stages[name] = elapsed
        print(f"  [{name}] {fmt_time(elapsed)}", flush=True)
        return result

    @property
    def total(self):
        return sum(self.stages.values())


# ---------------------------------------------------------------------------
# Backend 1: the colmap command line binary
# ---------------------------------------------------------------------------
class CliBackend:
    name = "colmap CLI"

    def __init__(self, exe, use_gpu, max_image_size):
        self.exe = exe
        self.use_gpu = use_gpu
        self.max_image_size = max_image_size

    @property
    def device(self):
        return "GPU" if self.use_gpu else "CPU"

    def _run(self, args, allow_gpu_retry=False):
        cmd = [self.exe] + args
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if allow_gpu_retry and self.use_gpu:
                # macOS builds frequently have no usable GPU SIFT context.
                print("      GPU path failed, retrying on CPU...", flush=True)
                self.use_gpu = False
                retry = self._swap_gpu_off(args)
                proc = subprocess.run([self.exe] + retry,
                                      capture_output=True, text=True)
                if proc.returncode == 0:
                    return proc
            sys.exit("\nCOLMAP failed:\n  " + " ".join(cmd) + "\n\n"
                     + (proc.stderr or proc.stdout)[-2500:])
        return proc

    @staticmethod
    def _swap_gpu_off(args):
        out = list(args)
        for i, a in enumerate(out):
            if a.endswith(".use_gpu") and i + 1 < len(out):
                out[i + 1] = "0"
        return out

    def extract(self, db, image_dir, single_camera):
        gpu = "1" if self.use_gpu else "0"
        self._run([
            "feature_extractor",
            "--database_path", db,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--ImageReader.camera_model", CAMERA_MODEL,
            "--SiftExtraction.max_num_features", str(N_FEATURES),
            "--SiftExtraction.max_image_size", str(self.max_image_size),
            "--SiftExtraction.use_gpu", gpu,
        ], allow_gpu_retry=True)

    def match(self, db, mode, vocab_tree):
        gpu = "1" if self.use_gpu else "0"
        if mode == "sequential":
            args = ["sequential_matcher",
                    "--database_path", db,
                    "--SequentialMatching.overlap", str(SEQ_OVERLAP),
                    "--SiftMatching.use_gpu", gpu]
            if vocab_tree:
                args += ["--SequentialMatching.loop_detection", "1",
                         "--SequentialMatching.vocab_tree_path", vocab_tree]
        else:
            args = ["exhaustive_matcher",
                    "--database_path", db,
                    "--SiftMatching.use_gpu", gpu]
        self._run(args, allow_gpu_retry=True)

    def map(self, db, image_dir, sparse_dir):
        os.makedirs(sparse_dir, exist_ok=True)
        self._run([
            "mapper",
            "--database_path", db,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
        ])

    def to_text(self, model_dir, txt_dir):
        os.makedirs(txt_dir, exist_ok=True)
        self._run([
            "model_converter",
            "--input_path", model_dir,
            "--output_path", txt_dir,
            "--output_type", "TXT",
        ])


# ---------------------------------------------------------------------------
# Backend 2: the pycolmap module
# ---------------------------------------------------------------------------
class PyColmapBackend:
    name = "pycolmap"

    def __init__(self, module, verbose, max_image_size):
        self.pc = module
        self.name = f"pycolmap {getattr(module, '__version__', '?')}"
        self.max_image_size = max_image_size
        # This wheel only has GPU SIFT if it was built with CUDA.
        self.device = "GPU" if getattr(module, "has_cuda", False) else "CPU"
        if not verbose:
            try:
                module.logging.minloglevel = module.logging.Level.WARNING
            except AttributeError:
                pass

    def extract(self, db, image_dir, single_camera):
        pc = self.pc
        kwargs = {"camera_mode": (pc.CameraMode.SINGLE if single_camera
                                  else pc.CameraMode.AUTO)}
        reader = pc.ImageReaderOptions()
        reader.camera_model = CAMERA_MODEL
        kwargs["reader_options"] = reader
        if hasattr(pc, "FeatureExtractionOptions"):          # pycolmap 4.x
            opts = pc.FeatureExtractionOptions()
            opts.sift.max_num_features = N_FEATURES
            opts.max_image_size = self.max_image_size
            kwargs["extraction_options"] = opts
        else:                                                # pycolmap 3.x
            opts = pc.SiftExtractionOptions()
            opts.max_num_features = N_FEATURES
            opts.max_image_size = self.max_image_size
            kwargs["sift_options"] = opts
        self.device = "GPU" if getattr(opts, "use_gpu", False) else "CPU"
        pc.extract_features(db, image_dir, **kwargs)

    def match(self, db, mode, vocab_tree):
        pc = self.pc
        if mode != "sequential":
            pc.match_exhaustive(db)
            return
        if hasattr(pc, "SequentialPairingOptions"):
            pairing = pc.SequentialPairingOptions()
            pairing.overlap = SEQ_OVERLAP
            if vocab_tree:
                pairing.loop_detection = True
                pairing.vocab_tree_path = vocab_tree
            pc.match_sequential(db, pairing_options=pairing)
        else:
            pc.match_sequential(db)

    def map(self, db, image_dir, sparse_dir):
        os.makedirs(sparse_dir, exist_ok=True)
        recs = self.pc.incremental_mapping(db, image_dir, sparse_dir)
        # 4.x writes the models itself; older versions hand them back in memory.
        if recs and not find_models(sparse_dir):
            items = recs.items() if hasattr(recs, "items") else enumerate(recs)
            for idx, rec in items:
                out = os.path.join(sparse_dir, str(idx))
                os.makedirs(out, exist_ok=True)
                rec.write(out)

    def to_text(self, model_dir, txt_dir):
        os.makedirs(txt_dir, exist_ok=True)
        self.pc.Reconstruction(model_dir).write_text(txt_dir)


def pick_backend(prefer, use_gpu, verbose, max_image_size):
    exe = shutil.which("colmap")
    if prefer in ("auto", "cli") and exe:
        return CliBackend(exe, use_gpu, max_image_size)
    if prefer in ("auto", "pycolmap"):
        try:
            import pycolmap
            return PyColmapBackend(pycolmap, verbose, max_image_size)
        except ImportError:
            pass
    if prefer == "cli" and not exe:
        sys.exit("--backend cli requested but `colmap` is not on PATH.")
    sys.exit(
        "Neither backend is available. Install one of:\n"
        "    brew install colmap          # the full COLMAP binary\n"
        "    pip install pycolmap         # same engine, python wheel\n")


# ---------------------------------------------------------------------------
# Reading COLMAP's outputs
# ---------------------------------------------------------------------------
def database_stats(db_path):
    """Feature counts and two-view geometry inliers, straight from the DB.

    two_view_geometries.rows is the number of RANSAC-verified inlier matches
    for that pair - the same quantity check_photos.py calls `inliers`.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    names = dict(cur.execute("SELECT image_id, name FROM images"))
    kps = dict(cur.execute("SELECT image_id, rows FROM keypoints"))

    pairs = []
    for pair_id, rows, config in cur.execute(
            "SELECT pair_id, rows, config FROM two_view_geometries"):
        if not rows:
            continue
        id2 = pair_id % MAX_PAIR_ID
        id1 = (pair_id - id2) // MAX_PAIR_ID
        pairs.append({
            "image1": names.get(id1, str(id1)),
            "image2": names.get(id2, str(id2)),
            "inliers": rows,
            "config": config,
        })
    con.close()

    feat = sorted(kps.values())
    inl = sorted(p["inliers"] for p in pairs)
    return {
        "features_per_image_median": int(statistics.median(feat)) if feat else 0,
        "features_per_image_min": feat[0] if feat else 0,
        "verified_pairs": len(pairs),
        "inliers_median": int(statistics.median(inl)) if inl else 0,
        "inliers_min": inl[0] if inl else 0,
        "inliers_max": inl[-1] if inl else 0,
        "pairs": pairs,
    }


def read_text_model(txt_dir):
    """Parse cameras.txt / images.txt / points3D.txt into benchmark numbers."""
    images, points, cameras = {}, {}, {}

    with open(os.path.join(txt_dir, "cameras.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            cameras[int(f[0])] = {
                "model": f[1],
                "width": int(f[2]),
                "height": int(f[3]),
                "params": [float(x) for x in f[4:]],
            }

    with open(os.path.join(txt_dir, "images.txt")) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        i += 1
        if not header:
            continue
        pts = lines[i].split() if i < len(lines) else []
        i += 1
        # POINTS2D triples; a POINT3D_ID of -1 means the keypoint is unused.
        observed = sum(1 for k in range(2, len(pts), 3) if pts[k] != "-1")
        images[int(header[0])] = {
            "name": header[9],
            "camera_id": int(header[8]),
            "observations": observed,
            "keypoints": len(pts) // 3,
        }

    errors, track_lengths = [], []
    with open(os.path.join(txt_dir, "points3D.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            error = float(f[7])
            track = (len(f) - 8) // 2
            errors.append(error)
            track_lengths.append(track)
            points[int(f[0])] = (error, track)

    n_obs = sum(track_lengths)
    # COLMAP's model_analyzer weights each point's error by its track length.
    mean_reproj = (sum(e * t for e, t in zip(errors, track_lengths)) / n_obs
                   if n_obs else 0.0)

    obs_per_image = [im["observations"] for im in images.values()]
    return {
        "registered_images": len(images),
        "points3D": len(points),
        "observations": n_obs,
        "mean_track_length": n_obs / len(points) if points else 0.0,
        "mean_observations_per_image": (n_obs / len(images)) if images else 0.0,
        "min_observations_per_image": min(obs_per_image) if obs_per_image else 0,
        "mean_reprojection_error_px": mean_reproj,
        "cameras": cameras,
        "image_names": sorted((im["name"] for im in images.values()),
                              key=natural_key),
    }


def find_models(sparse_dir):
    if not os.path.isdir(sparse_dir):
        return []
    out = []
    for entry in sorted(os.listdir(sparse_dir), key=natural_key):
        if entry.endswith("_yup"):      # viewing copy of another model, not a model
            continue
        path = os.path.join(sparse_dir, entry)
        if os.path.isdir(path) and (
                os.path.exists(os.path.join(path, "images.bin"))
                or os.path.exists(os.path.join(path, "images.txt"))):
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(folder, names, models, stats, watch, backend, settings, out_json):
    backend_name = f"{backend.name} ({backend.device} SIFT)"
    n_total = len(names)
    best = max(models, key=lambda m: m["registered_images"])
    frac = best["registered_images"] / n_total

    print("\n" + "=" * 68)
    print(f"COLMAP BENCHMARK  -  {os.path.basename(os.path.abspath(folder))}")
    print("=" * 68)

    print(f"\nbackend                     : {backend_name}")
    print(f"matcher                     : {settings['matcher']}")
    print(f"max image size / keypoints  : {settings['max_image_size']} px"
          f" / {N_FEATURES}")
    print(f"input images                : {n_total}")
    print(f"models reconstructed        : {len(models)}"
          + ("  (should be 1 - the set split apart)" if len(models) > 1 else ""))

    print("\n-- largest model " + "-" * 51)
    print(f"registered images           : {best['registered_images']} / "
          f"{n_total}   ({frac * 100:.0f}%)")
    print(f"sparse 3D points            : {best['points3D']}")
    print(f"observations (2D-3D links)  : {best['observations']}")
    print(f"mean track length           : {best['mean_track_length']:.2f}"
          "   (images seeing each point)")
    print(f"mean observations / image   : "
          f"{best['mean_observations_per_image']:.0f}")
    print(f"min observations / image    : {best['min_observations_per_image']}")
    print(f"mean reprojection error     : "
          f"{best['mean_reprojection_error_px']:.3f} px")
    for cam in best["cameras"].values():
        print(f"camera                      : {cam['model']} "
              f"{cam['width']}x{cam['height']}  f={cam['params'][0]:.1f} px")
        break

    print("\n-- two-view geometry (COLMAP's own RANSAC) " + "-" * 25)
    print(f"descriptors / image (median): "
          f"{stats['features_per_image_median']}"
          f"   (from a {N_FEATURES}-keypoint cap)")
    print(f"verified pairs              : {stats['verified_pairs']}")
    print(f"inliers per pair            : median "
          f"{stats['inliers_median']}, min {stats['inliers_min']}, "
          f"max {stats['inliers_max']}")

    print("\n-- wall clock " + "-" * 54)
    if watch.stages:
        for stage, secs in watch.stages.items():
            print(f"{stage:<28}: {fmt_time(secs)}")
        print(f"{'TOTAL':<28}: {fmt_time(watch.total)}"
              f"   ({watch.total / n_total:.1f}s per image)")
    else:
        print("reported from an existing workspace - nothing was recomputed")

    missing = sorted(set(names) - set(best["image_names"]), key=natural_key)
    if missing:
        print(f"\ndropped images ({len(missing)}):")
        print("   " + ", ".join(missing))

    print()
    if len(models) > 1:
        print("VERDICT: SPLIT. COLMAP built more than one disconnected model.")
        print("   The photo chain breaks somewhere - the dropped images above")
        print("   are the place to look. Reshoot the gap and rerun.")
    elif frac >= GOOD_REGISTERED:
        print("VERDICT: GOOD. One connected model, nearly every photo placed.")
    elif frac >= WEAK_REGISTERED:
        print("VERDICT: PARTIAL. One model, but some photos never registered.")
        print("   Usable, though the dropped views contribute nothing.")
    else:
        print("VERDICT: POOR. Most photos failed to register.")
        print("   Treat this set as not reconstructable without more coverage.")

    summary = {
        "folder": os.path.abspath(folder),
        "backend": backend_name,
        "settings": settings,
        "input_images": n_total,
        "models_reconstructed": len(models),
        "largest_model": {k: v for k, v in best.items()
                          if k not in ("cameras", "image_names")},
        "registered_fraction": frac,
        "dropped_images": missing,
        "two_view_geometry": {k: v for k, v in stats.items() if k != "pairs"},
        "timing_seconds": dict(watch.stages, TOTAL=watch.total),
    }
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nmachine-readable summary -> {out_json}")
    print()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run COLMAP on a photo folder and print benchmark numbers.")
    ap.add_argument("folder")
    ap.add_argument("--workspace",
                    help="where to put database/ and sparse/ "
                         "(default: colmap_out/<folder name>)")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"],
                    default="exhaustive",
                    help="exhaustive is the benchmark default; sequential is "
                         "much faster on a long walk-around sequence")
    ap.add_argument("--vocab-tree", metavar="PATH",
                    help="sequential matcher only: vocabulary tree file, "
                         "which turns on loop detection (download a "
                         "vocab_tree_flickr100K_*.bin from demuc.de/colmap/)")
    ap.add_argument("--max-image-size", type=int, default=MAX_IMAGE_SIZE,
                    metavar="PX",
                    help=f"downscale the long edge before SIFT "
                         f"(default {MAX_IMAGE_SIZE}; feature extraction "
                         f"dominates the wall clock, so this is the main "
                         f"speed lever)")
    ap.add_argument("--multi-camera", action="store_true",
                    help="photos came from different cameras / zoom levels")
    ap.add_argument("--backend", choices=["auto", "cli", "pycolmap"],
                    default="auto")
    ap.add_argument("--no-gpu", action="store_true",
                    help="force CPU SIFT (CLI backend)")
    ap.add_argument("--verbose", action="store_true",
                    help="show COLMAP's own progress logging")
    ap.add_argument("--force", action="store_true",
                    help="delete an existing workspace and start clean")
    ap.add_argument("--reuse", action="store_true",
                    help="skip any stage whose output already exists")
    ap.add_argument("--no-yup", action="store_true",
                    help="don't build sparse/0_yup, the PLYs and preview.jpg")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a directory: {args.folder}")
    names = list_images(args.folder)
    if len(names) < 3:
        sys.exit(f"Need at least 3 images, found {len(names)}")

    tag = os.path.basename(os.path.abspath(args.folder))
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workspace = args.workspace or os.path.join(_root, "results", "colmap_out", tag)
    if os.path.exists(workspace) and not (args.reuse or args.force):
        sys.exit(f"Workspace {workspace} already exists.\n"
                 "  --reuse to re-report from it, --force to rebuild it.")
    if args.force and os.path.exists(workspace):
        print(f"Removing existing workspace {workspace}")
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)

    db = os.path.join(workspace, "database.db")
    sparse = os.path.join(workspace, "sparse")
    image_dir = os.path.abspath(args.folder)

    backend = pick_backend(args.backend, use_gpu=not args.no_gpu,
                           verbose=args.verbose,
                           max_image_size=args.max_image_size)

    print(f"\n{len(names)} images in {args.folder}")
    print(f"backend   : {backend.name}")
    print(f"workspace : {workspace}")
    loop_note = " + loop detection" if args.vocab_tree else ""
    print(f"matcher   : {args.matcher}{loop_note}\n")
    if args.vocab_tree and args.matcher != "sequential":
        print("  note: --vocab-tree only applies to --matcher sequential\n")
    if args.vocab_tree and not os.path.exists(args.vocab_tree):
        sys.exit(f"Vocabulary tree not found: {args.vocab_tree}")

    watch = Stopwatch()

    if args.reuse and os.path.exists(db):
        print("  [features] reusing existing database")
        print("  [matching] reusing existing database")
    else:
        watch.run("features", lambda: backend.extract(
            db, image_dir, single_camera=not args.multi_camera))
        watch.run("matching", lambda: backend.match(
            db, args.matcher, args.vocab_tree))

    if args.reuse and find_models(sparse):
        print("  [mapping]  reusing existing sparse model")
    else:
        watch.run("mapping", lambda: backend.map(db, image_dir, sparse))

    model_dirs = find_models(sparse)
    if not model_dirs:
        sys.exit("\nCOLMAP produced no reconstruction at all. The photo set "
                 "has no usable geometry - rerun check_photos.py and fix the "
                 "BROKEN pairs first.")

    models = []
    for model_dir in model_dirs:
        txt_dir = os.path.join(model_dir, "txt")
        if not os.path.exists(os.path.join(txt_dir, "images.txt")):
            backend.to_text(model_dir, txt_dir)
        models.append(read_text_model(txt_dir))

    stats = database_stats(db)
    settings = {
        "matcher": args.matcher,
        "max_image_size": args.max_image_size,
        "max_num_features": N_FEATURES,
        "camera_model": CAMERA_MODEL,
        "single_camera": not args.multi_camera,
        "loop_detection": bool(args.vocab_tree),
    }
    report(args.folder, names, models, stats, watch, backend, settings,
           os.path.join(workspace, "benchmark.json"))

    if not args.no_yup:
        make_viewing_copy(workspace)


def make_viewing_copy(workspace):
    """sparse/0 -> sparse/0_raw, plus 0_yup, the PLYs and preview.jpg."""
    if os.path.isdir(os.path.join(workspace, "sparse", "0_yup")):
        print("  [yup] sparse/0_yup already exists - left as is")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_yup
    print("  [yup] building the gravity = +Y viewing copy...")
    try:
        make_yup.process(workspace)
    except RuntimeError as e:
        print(f"  [yup] skipped: {e}")
    print()


if __name__ == "__main__":
    main()
