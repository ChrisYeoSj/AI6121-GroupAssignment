#!/usr/bin/env python3
"""
selftest.py - prove the fixture is self-consistent. Run after regenerating.

    python synthetic/selftest.py

Also doubles as a worked example of reading the contract.
"""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_synthetic import load

fails = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok: fails.append(name)

S = load("clean")
K, n = S.K, S.n_cameras

# 1. GT poses reproject GT points exactly onto the stored keypoints
worst = 0.0
for i in range(n):
    kp, _, _ = S.features[i]; pid = S.gt_kp_point_id[i]; m = pid >= 0
    rvec, tvec = S.gt_poses[i]
    pr, _ = cv2.projectPoints(S.gt_xyz[pid[m]], rvec, tvec, K, np.zeros(5))
    worst = max(worst, float(np.abs(pr.reshape(-1, 2) - kp[m]).max()))
check("GT poses reproject GT points onto stored keypoints", worst < 1e-2,
      f"max {worst:.2e} px")

# 2. camera centre convention: C == -R.T @ tvec
d = max(float(np.linalg.norm(S.gt_centres[i] +
        cv2.Rodrigues(S.gt_poses[i][0])[0].T @ S.gt_poses[i][1])) for i in range(n))
check("gt_centres == -R.T @ tvec  (tvec is translation)", d < 1e-6, f"max {d:.2e} mm")

# 3. every match in 'clean' joins keypoints of the SAME 3D point
bad = 0; tot = 0
for (i, j), m in S.matches.items():
    a = S.gt_kp_point_id[i][m[:, 0]]; b = S.gt_kp_point_id[j][m[:, 1]]
    bad += int((a != b).sum()); tot += len(m)
check("clean matches all join the same 3D point", bad == 0, f"{bad}/{tot} wrong")

# 4. a match never pairs a decoy
dec = sum(int((S.gt_kp_point_id[i][m[:, 0]] < 0).sum()) for (i, j), m in S.matches.items())
check("no match involves a decoy keypoint", dec == 0, f"{dec} found")

# 5. triangulating from GT poses recovers GT xyz
errs = []
for (i, j) in S.pairs("general")[:20]:
    m = S.matches[(i, j)]
    Ri = cv2.Rodrigues(S.gt_poses[i][0])[0]; Rj = cv2.Rodrigues(S.gt_poses[j][0])[0]
    P1 = K @ np.hstack([Ri, S.gt_poses[i][1].reshape(3, 1)])
    P2 = K @ np.hstack([Rj, S.gt_poses[j][1].reshape(3, 1)])
    x1 = S.features[i][0][m[:, 0]].T.astype(np.float64)
    x2 = S.features[j][0][m[:, 1]].T.astype(np.float64)
    X = cv2.triangulatePoints(P1, P2, x1, x2); X = (X[:3] / X[3]).T
    errs.append(np.linalg.norm(X - S.gt_xyz[S.gt_kp_point_id[i][m[:, 0]]], axis=1))
e = np.concatenate(errs)
check("triangulation from GT poses recovers GT xyz", float(np.median(e)) < 1e-3,
      f"median {np.median(e):.2e} mm")

# 6. panoramic pairs really have zero baseline
pano = S.pairs("panoramic")
bl = [float(np.linalg.norm(S.gt_centres[i] - S.gt_centres[j])) for i, j in pano]
check("panoramic pairs have zero baseline", len(pano) > 0 and max(bl) < 1e-9,
      f"{len(pano)} pairs, max baseline {max(bl):.2e} mm")

# 7. planar pairs really are coplanar
worst_plane = 0.0
for (i, j) in S.pairs("planar"):
    ids = S.gt_kp_point_id[i][S.matches[(i, j)][:, 0]]
    P = S.gt_xyz[ids]; Pc = P - P.mean(0)
    sv = np.linalg.svd(Pc, compute_uv=False)
    worst_plane = max(worst_plane, float(sv[2] / sv[0]))     # thickness / extent
check("planar pairs are coplanar", worst_plane < 1e-6, f"max thickness ratio {worst_plane:.2e}")

# 8. rejected pairs are below the plan's N_F_MIN = 30
rej = S.pairs("rejected")
mx = max(len(S.matches[p]) for p in rej) if rej else 0
check("rejected pairs have < 30 matches", mx < 30, f"{len(rej)} pairs, max {mx}")

# 9. outlier variant corrupts exactly 30%
O = load("outliers")
tot = sum(len(v) for v in O.gt_match_inlier.values())
good = sum(int(v.sum()) for v in O.gt_match_inlier.values())
frac = 1 - good / tot
check("outliers variant is 30% corrupted", abs(frac - 0.30) < 0.01, f"{frac*100:.1f}%")

# 10. flagged outlier rows really are wrong correspondences
wrong_ok = True
for (i, j), m in O.matches.items():
    inl = O.gt_match_inlier[(i, j)]
    if inl.all(): continue
    a = O.gt_kp_point_id[i][m[~inl, 0]]; b = O.gt_kp_point_id[j][m[~inl, 1]]
    if (a == b).any(): wrong_ok = False; break
check("rows flagged as outliers are genuinely wrong", wrong_ok)

# 11. noisy variant noise level
N = load("noisy")
r = []
for i in range(n):
    pid = N.gt_kp_point_id[i]; m = pid >= 0
    pr, _ = cv2.projectPoints(N.gt_xyz[pid[m]], *N.gt_poses[i], K, np.zeros(5))
    r.append(np.linalg.norm(pr.reshape(-1, 2) - N.features[i][0][m], axis=1))
rms = float(np.sqrt((np.concatenate(r) ** 2).mean()))
check("noisy variant has ~1.41 px radial RMS", abs(rms - 1.414) < 0.1, f"{rms:.3f} px")

print("\n  %d/%d checks passed" % (11 - len(fails), 11))
sys.exit(1 if fails else 0)
