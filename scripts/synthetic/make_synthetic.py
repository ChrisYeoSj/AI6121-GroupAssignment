#!/usr/bin/env python3
"""
make_synthetic.py - build the synthetic test scene the plan asks for in Week 1.

WHY THIS EXISTS
    "Week 1, everyone, before any real photo exists: build the synthetic test
    scene. Build it at full size straight away - 30-40 camera poses and a few
    thousand 3D points, not a 5-camera toy." (plan, Part 7)

    Every downstream stage can be implemented and unit-tested against this
    before a single photograph is matched, because here you know the answer.

WHAT IT PRODUCES
    Three variants, exactly as the plan specifies:
      clean     - perfect projections, correct correspondences
      noisy     - Gaussian pixel noise on every observation
      outliers  - 30% of correspondences are deliberately wrong

    Stage 2 (feature extraction) is bypassed: descriptors are synthesised, so
    Stage 3 onwards can run unmodified.

GEOMETRY - deliberately built to exercise Stage 3b's whole decision tree
    30 loop cameras   a full 360 deg orbit, 12 deg apart -> 'general' pairs
     3 rotation-only  same camera centre as cam 00, rotated -> 'panoramic'
     4 wall cameras   viewing a flat target only          -> 'planar'
    A scene graph with only 'general' edges never exercises the branches that
    the paper's first contribution is entirely about.

CONVENTIONS (match the Part 7 data contract exactly)
    X_cam = R @ X_world + t        so tvec is TRANSLATION, not camera position
    dist  = zeros(5)               the pipeline works on UNDISTORTED images;
                                   K is the matrix they were undistorted with
    matches are integer keypoint indices, never coordinates
"""
import argparse, json, os
import numpy as np
import cv2

N_LOOP, N_PANO, N_WALL = 30, 3, 4
N_ROCK_PTS, N_WALL_PTS = 7000, 1600
DECOYS_PER_IMAGE = 120          # keypoints matching no 3D point, so the ratio
                                # test and mutual check have something to reject
MAX_VIEW_DEG = 42.0             # descriptor viewpoint tolerance (see project loop)
NOISE_SIGMA_PX = 1.0
OUTLIER_FRAC = 0.30
N_F_MIN = 30                    # plan: hard floor for admitting an edge
ALPHA_PANO_DEG = 2.0            # plan: below this a degenerate pair is panoramic


def rock_points(rng, n):
    """Star-shaped vertical blob: radius modulated by smooth noise.

    Star-shaped means the outward radial direction is a usable surface normal,
    which is what lets us model self-occlusion cheaply below.
    """
    phi = rng.uniform(0, 2 * np.pi, n)
    ct = rng.uniform(-1, 1, n)
    theta = np.arccos(ct)
    bump = (1.0 + 0.22 * np.sin(3 * phi) * np.sin(2 * theta)
                + 0.12 * np.sin(7 * phi + 1.3) + 0.08 * rng.normal(size=n))
    r = 700.0 * bump                                   # mm
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi) * 0.75
    z = r * np.cos(theta) * 1.7                        # elongated vertically
    P = np.stack([x, y, z], 1)
    N = P / np.linalg.norm(P, axis=1, keepdims=True)   # outward normal
    return P, N


def wall_points(rng, n):
    """A flat target set well away from the rock -> genuinely planar pairs."""
    u = rng.uniform(-900, 900, n)      # in-plane axis 1  -> world y
    v = rng.uniform(-600, 600, n)      # in-plane axis 2  -> world z
    P = np.stack([np.full(n, 5200.0), u, v], 1)        # the plane x = 5200
    N = np.tile(np.array([-1.0, 0, 0]), (n, 1))        # faces back toward origin
    return P, N


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """World->camera rotation for a camera at `eye` looking at `target`."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, -u, f])          # rows: x_cam, y_cam, z_cam
    return R


def build_cameras(rng):
    """Returns (R_list, t_list, centres, kind_list)."""
    Rs, ts, Cs, kinds = [], [], [], []
    # --- 30 loop cameras: full orbit, 12 deg apart -----------------------
    for i in range(N_LOOP):
        a = 2 * np.pi * i / N_LOOP
        C = np.array([3000 * np.cos(a), 3000 * np.sin(a), 350 + 120 * np.sin(2 * a)])
        R = look_at(C, np.array([0.0, 0.0, 0.0]))
        Rs.append(R); Cs.append(C); ts.append(-R @ C); kinds.append("loop")
    # --- 3 rotation-only cameras: IDENTICAL centre to cam 00 -------------
    C0 = Cs[0]
    for k in range(N_PANO):
        ang = np.deg2rad(4.0 * (k + 1))
        Rz = cv2.Rodrigues(np.array([0.0, 0.0, ang]))[0]
        R = Rs[0] @ Rz.T
        Rs.append(R); Cs.append(C0.copy()); ts.append(-R @ C0); kinds.append("pano")
    # --- 4 wall cameras: see the flat target only ------------------------
    wall_ctr = np.array([5200.0, 0.0, 0.0])
    for k in range(N_WALL):
        C = wall_ctr + np.array([-2200.0, -700 + 470 * k, 200 * ((k % 2) - .5)])
        R = look_at(C, wall_ctr + np.array([0.0, -300 + 200 * k, 0.0]))
        Rs.append(R); Cs.append(C); ts.append(-R @ C); kinds.append("wall")
    return Rs, ts, Cs, kinds


def rootsift(v):
    v = np.abs(v)
    v = v / np.maximum(v.sum(axis=-1, keepdims=True), 1e-12)
    return np.sqrt(v).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Build the synthetic SfM test scene.")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _root = os.path.dirname(_root)                      # repo root
    ap.add_argument("--out", default=os.path.join(_root, "results", "synthetic"))
    ap.add_argument("--calib",
                    default=os.path.join(_root, "results", "calibration",
                                         "calibration.json"),
                    help="take K and the resolution from a real calibration")
    ap.add_argument("--seed", type=int, default=6121)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- intrinsics: real K, but ZERO distortion (pipeline sees undistorted)
    cal = json.load(open(args.calib))
    K = np.array(cal["K"], float)
    W, H = int(cal["resolution"][0]), int(cal["resolution"][1])
    dist = np.zeros(5)

    # ---- scene ---------------------------------------------------------
    Pr, Nr = rock_points(rng, N_ROCK_PTS)
    Pw, Nw = wall_points(rng, N_WALL_PTS)
    xyz = np.vstack([Pr, Pw]).astype(np.float64)
    nrm = np.vstack([Nr, Nw])
    is_wall = np.r_[np.zeros(len(Pr), bool), np.ones(len(Pw), bool)]
    rgb = np.clip(rng.normal([95, 92, 90], 28, (len(xyz), 3)), 0, 255).astype(np.uint8)
    rgb[is_wall] = np.clip(rng.normal([170, 165, 155], 18,
                                      (is_wall.sum(), 3)), 0, 255).astype(np.uint8)
    base_desc = rootsift(rng.gamma(0.7, 1.0, (len(xyz), 128)))

    Rs, ts, Cs, kinds = build_cameras(rng)
    n_cam = len(Rs)

    # ---- project, with facing-away self-occlusion -----------------------
    kpts, ptid, desc, cols = [], [], [], []
    for i in range(n_cam):
        R, t, C = Rs[i], ts[i], Cs[i]
        Xc = (R @ xyz.T).T + t
        infront = Xc[:, 2] > 1.0
        # A descriptor stops matching long before the surface turns edge-on:
        # SIFT tolerates roughly 40-50 deg of viewpoint change. Using the bare
        # facing test (< 90 deg) let every point be seen by ~half the orbit and
        # produced median track length 12, where COLMAP measures 4.86 on the
        # real scene. Cap the viewing angle instead.
        vdir = C - xyz
        vdir = vdir / np.linalg.norm(vdir, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", nrm, vdir) > np.cos(np.deg2rad(MAX_VIEW_DEG))
        proj = (K @ Xc.T).T
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
        inside = (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
        vis = infront & facing & inside
        idx = np.flatnonzero(vis)
        rng.shuffle(idx)
        k_uv = uv[idx].astype(np.float32)
        k_id = idx.astype(np.int32)
        k_d = base_desc[idx] + rng.normal(0, 0.006, (len(idx), 128)).astype(np.float32)
        k_d = (k_d / np.linalg.norm(k_d, axis=1, keepdims=True)).astype(np.float32)
        k_c = rgb[idx]
        # decoys: real keypoints that correspond to no 3D point
        d_uv = np.stack([rng.uniform(5, W - 5, DECOYS_PER_IMAGE),
                         rng.uniform(5, H - 5, DECOYS_PER_IMAGE)], 1).astype(np.float32)
        d_d = rootsift(rng.gamma(0.7, 1.0, (DECOYS_PER_IMAGE, 128)))
        kpts.append(np.vstack([k_uv, d_uv]).astype(np.float32))
        ptid.append(np.r_[k_id, np.full(DECOYS_PER_IMAGE, -1, np.int32)].astype(np.int32))
        desc.append(np.vstack([k_d, d_d]).astype(np.float32))
        cols.append(np.vstack([k_c, rng.integers(0, 255, (DECOYS_PER_IMAGE, 3),
                                                 dtype=np.uint8)]).astype(np.uint8))

    # ---- ground-truth correspondences + pair labels ---------------------
    where = [{p: k for k, p in enumerate(ptid[i]) if p >= 0} for i in range(n_cam)]
    pairs = {}
    for i in range(n_cam):
        for j in range(i + 1, n_cam):
            shared = np.array(sorted(set(where[i]) & set(where[j])), dtype=np.int64)
            if len(shared) == 0:
                continue
            m = np.stack([[where[i][p] for p in shared],
                          [where[j][p] for p in shared]], 1).astype(np.int32)
            # GT median triangulation angle, at the 3D points themselves
            v1 = Cs[i] - xyz[shared]; v2 = Cs[j] - xyz[shared]
            v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
            v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
            ang = float(np.median(np.degrees(np.arccos(
                np.clip((v1 * v2).sum(1), -1, 1)))))
            base = float(np.linalg.norm(Cs[i] - Cs[j]))
            if len(m) < N_F_MIN:
                label = "rejected"
            elif base < 1e-6 or ang < ALPHA_PANO_DEG:
                label = "panoramic"
            elif is_wall[shared].all():
                label = "planar"
            else:
                label = "general"
            pairs[(i, j)] = dict(m=m, alpha=ang, label=label, shared=shared)

    # ---- write the three variants ---------------------------------------
    manifest = {}
    for variant in ("clean", "noisy", "outliers"):
        vr = np.random.default_rng(args.seed + hash(variant) % 10000)
        blob = {"K": K, "dist": dist, "calib_resolution": np.array([W, H]),
                "n_cameras": n_cam,
                "gt_rvecs": np.array([cv2.Rodrigues(R)[0].ravel() for R in Rs]),
                "gt_tvecs": np.array(ts),
                "gt_centres": np.array(Cs),
                "gt_xyz": xyz, "gt_rgb": rgb,
                "gt_is_wall": is_wall,
                "camera_kind": np.array(kinds),
                "variant": variant}
        for i in range(n_cam):
            kp = kpts[i].copy()
            if variant in ("noisy", "outliers"):
                kp += vr.normal(0, NOISE_SIGMA_PX, kp.shape).astype(np.float32)
            blob[f"kpts_{i:02d}"] = kp
            blob[f"desc_{i:02d}"] = desc[i]
            blob[f"colors_{i:02d}"] = cols[i]
            blob[f"ptid_{i:02d}"] = ptid[i]
        n_corrupt = 0
        for (i, j), d in pairs.items():
            m = d["m"].copy()
            inl = np.ones(len(m), bool)
            if variant == "outliers" and len(m) >= 4:
                k = int(round(OUTLIER_FRAC * len(m)))
                sel = vr.choice(len(m), k, replace=False)
                true_pt = ptid[j][m[sel, 1]]
                cand = vr.integers(0, len(kpts[j]), k)
                # A uniformly random partner lands on the CORRECT keypoint about
                # once per 1000 rows. Resample those, or the inlier mask lies.
                for _ in range(8):
                    clash = ptid[j][cand] == true_pt
                    if not clash.any():
                        break
                    cand[clash] = vr.integers(0, len(kpts[j]), int(clash.sum()))
                m[sel, 1] = cand
                n_corrupt += k
            # Derive the mask from ground truth rather than from intent, so it
            # is true by construction whatever the sampling did.
            a = ptid[i][m[:, 0]]
            b = ptid[j][m[:, 1]]
            inl = (a == b) & (a >= 0)
            blob[f"m_{i:02d}_{j:02d}"] = m
            blob[f"minl_{i:02d}_{j:02d}"] = inl
            blob[f"lab_{i:02d}_{j:02d}"] = np.array(
                [d["label"], f"{d['alpha']:.4f}"], dtype=object)
        path = os.path.join(args.out, f"{variant}.npz")
        np.savez_compressed(path, **blob)
        manifest[variant] = dict(path=path, corrupted_matches=n_corrupt,
                                 size_mb=round(os.path.getsize(path) / 1e6, 1))
        print(f"  wrote {path}  ({manifest[variant]['size_mb']} MB)")

    labs = {}
    for d in pairs.values():
        labs[d["label"]] = labs.get(d["label"], 0) + 1
    meta = dict(seed=args.seed, n_cameras=n_cam, n_points=len(xyz),
                cameras=dict(loop=N_LOOP, panoramic=N_PANO, wall=N_WALL),
                pairs_with_any_overlap=len(pairs), label_counts=labs,
                noise_sigma_px=NOISE_SIGMA_PX, outlier_fraction=OUTLIER_FRAC,
                K=K.tolist(), resolution=[W, H], variants=manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n  cameras {n_cam} ({N_LOOP} loop / {N_PANO} pano / {N_WALL} wall)"
          f"   points {len(xyz)}")
    print(f"  pairs with overlap: {len(pairs)}   labels: {labs}")
    print(f"  wrote {os.path.join(args.out,'manifest.json')}")


if __name__ == "__main__":
    main()
