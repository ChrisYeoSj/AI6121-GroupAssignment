import math
import numpy as np
import config as C

DEGENERATE, UNCALIBRATED, PLANAR, PANORAMIC, PLANAR_OR_PANORAMIC = (
    "DEGENERATE", "UNCALIBRATED", "PLANAR", "PANORAMIC", "PLANAR_OR_PANORAMIC")


def homogeneous(p):
    h = np.ones((len(p), 3))
    h[:, :2] = p
    return h


def normalize(p):
    c = p.mean(0)
    rms = np.sqrt(((p - c) ** 2).sum(1).mean())
    s = np.sqrt(2.0) / rms if rms > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (p - c) * s, T


def sampson(F, x1, x2):
    l1 = x1 @ F[:, :2].T + F[:, 2]
    l2 = x2 @ F[:2, :2] + F[2, :2]
    num = (x2 * l1[:, :2]).sum(1) + l1[:, 2]
    den = l1[:, 0] ** 2 + l1[:, 1] ** 2 + l2[:, 0] ** 2 + l2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num * num / den
    return np.where(den == 0, np.finfo(float).max, r)


def transfer(H, x1, x2):
    q = x1 @ H[:, :2].T + H[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = x2 - q[:, :2] / q[:, 2:3]
    r = (d ** 2).sum(1)
    return np.where(np.isfinite(r), r, np.finfo(float).max)


class SevenPoint:
    k = 7

    def estimate(self, x1, x2):
        p1, p2 = homogeneous(x1), homogeneous(x2)
        A = np.stack([np.concatenate([a[0] * b, a[1] * b, b]) for a, b in zip(p1, p2)], 1)
        Q = np.linalg.qr(A, mode="complete")[0]
        g1, g2 = Q[:, 7], Q[:, 8]
        rs = np.array([0.0, 1.0, -1.0, 2.0])
        dets = [np.linalg.det((r * g1 + g2).reshape(3, 3, order="F")) for r in rs]
        c = np.linalg.solve(np.vander(rs, 4), dets)
        if abs(c[0]) < 1e-16:
            return []
        roots = np.roots(c)
        out = []
        for r in roots[np.isreal(roots)].real:
            f = r * g1 + g2
            out.append((f / np.linalg.norm(f)).reshape(3, 3, order="F"))
        return out

    def residuals(self, x1, x2, F):
        return sampson(F, x1, x2)


class EightPoint:
    k = 8

    def estimate(self, x1, x2):
        n1, T1 = normalize(x1)
        n2, T2 = normalize(x2)
        p1 = homogeneous(n1)
        A = np.c_[n2[:, :1] * p1, n2[:, 1:2] * p1, p1]
        Q = np.linalg.svd(A, full_matrices=len(A) < 9)[2][-1].reshape(3, 3)
        U, s, Vt = np.linalg.svd(Q)
        s[2] = 0.0
        return [T2.T @ (U * s) @ Vt @ T1]

    def residuals(self, x1, x2, F):
        return sampson(F, x1, x2)


def collinear_triplet(p):
    def col(i, j, k):
        d1, d2 = p[j] - p[i], p[k] - p[i]
        sc = (d1 @ d1) * (d2 @ d2)
        area = d1[0] * d2[1] - d1[1] * d2[0]
        return sc == 0.0 or area * area <= 1e-24 * sc
    return col(0, 1, 2) or col(0, 1, 3) or col(0, 2, 3) or col(1, 2, 3)


def cheirality_ok(x1, x2):
    a, b = homogeneous(x1), homogeneous(x2)
    p, q = np.cross(a[0], a[1]), np.cross(b[0], b[1])
    if p @ a[2] * (q @ b[2]) < 0 or p @ a[3] * (q @ b[3]) < 0:
        return False
    p, q = np.cross(a[2], a[3]), np.cross(b[2], b[3])
    if p @ a[0] * (q @ b[0]) < 0 or p @ a[1] * (q @ b[1]) < 0:
        return False
    return True


class Homography:
    k = 4

    def __init__(self, check_cheirality=False):
        self.check = check_cheirality

    def estimate(self, x1, x2):
        if len(x1) == 4:
            if self.check and not cheirality_ok(x1, x2):
                return []
            if collinear_triplet(x1) or collinear_triplet(x2):
                return []
        n1, T1 = normalize(x1)
        n2, T2 = normalize(x2)
        a, b = homogeneous(n1), homogeneous(n2)
        z = np.zeros_like(a)
        A = np.empty((2 * len(a), 9))
        A[0::2] = np.c_[a, z, -b[:, :1] * a]
        A[1::2] = np.c_[z, a, -b[:, 1:2] * a]
        _, s, Vt = np.linalg.svd(A, full_matrices=len(A) < 9)
        if s[7] <= 1e-12 * s[0]:
            return []
        H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
        sv = np.linalg.svd(H, compute_uv=False)
        if not (sv[0] > 0 and sv[2] > 1e-12 * sv[0]) or not np.all(np.isfinite(H)):
            return []
        return [H / np.linalg.norm(H)]

    def residuals(self, x1, x2, H):
        return transfer(H, x1, x2)


def msac(res, max_res):
    inl = res <= max_res
    return int(inl.sum()), float(np.where(inl, res, max_res).sum())


def num_trials(n_in, n, k, confidence=C.CONFIDENCE, mult=C.DYN_TRIALS_MULTIPLIER):
    fail = 1 - confidence
    p_in = 1.0
    for i in range(k):
        p_in *= max(n_in - i, 0) / (n - i)
    p_out = 1 - p_in
    if p_out <= 0:
        return 1
    if p_out == 1.0:
        return math.inf
    return math.ceil(math.log(fail) / math.log(p_out) * mult)


def loransac(x1, x2, est, local, rng, max_error=C.MAX_ERROR, min_inlier_ratio=C.MIN_INLIER_RATIO):
    n = len(x1)
    out = {"success": False, "model": None, "inliers": 0, "mask": None}
    if n < est.k:
        return out
    max_res = max_error * max_error
    big = 100000
    max_trials = min(C.MAX_NUM_TRIALS, num_trials(int(min_inlier_ratio * big), big, est.k))
    dyn_max = max_trials
    best_score, best_model, best_inl, best_local = math.inf, None, 0, False
    trial = 0
    while trial < max_trials:
        cur = trial
        trial += 1
        sample = rng.choice(n, est.k, replace=False)
        stop = False
        for m in est.estimate(x1[sample], x2[sample]):
            res = est.residuals(x1, x2, m)
            n_in, score = msac(res, max_res)
            if score < best_score:
                l_model, l_inl, l_score, l_local = m, n_in, score, False
                if n_in > est.k and n_in >= local.k:
                    for _ in range(C.LOCAL_TRIALS):
                        sel = res <= max_res
                        if sel.sum() < local.k:
                            break
                        improved = False
                        for lm in local.estimate(x1[sel], x2[sel]):
                            lres = local.residuals(x1, x2, lm)
                            li, ls = msac(lres, max_res)
                            if ls < l_score:
                                l_model, l_inl, l_score, l_local = lm, li, ls, True
                                best_res = lres
                                improved = True
                        if not improved:
                            break
                        res = best_res
                if l_score < best_score:
                    best_score, best_model, best_inl, best_local = l_score, l_model, l_inl, l_local
                    dyn_max = num_trials(best_inl, n, est.k)
            if cur >= dyn_max and cur >= C.MIN_NUM_TRIALS:
                stop = True
                break
        if stop:
            break
    if best_model is None:
        return out
    out["model"], out["inliers"] = best_model, best_inl
    if best_inl < est.k:
        return out
    out["success"] = True
    res = (local if best_local else est).residuals(x1, x2, best_model)
    out["mask"] = res <= max_res
    return out


def angle_between(a, b):
    sq = (a * a).sum(1) * (b * b).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.clip((a * b).sum(1) / np.sqrt(sq), -1.0, 1.0)
    return np.where(sq > 0, np.arccos(c), 0.0)


def fold(a):
    return np.minimum(a, np.pi - a)


def cheirality(R, t, r1, r2):
    r1c = r1 @ R.T
    a = -(r1c * r2).sum(1)
    b1 = -(r1c @ t)
    b2 = r2 @ t
    return (b1 - a * b2 > 0) & (b2 - a * b1 > 0)


def pose_from_essential(Em, r1, r2):
    U, _, V = np.linalg.svd(Em)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(V) < 0:
        V = -V
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    R1, R2 = U @ W @ V, U @ W.T @ V
    t = U[:, 2] / np.linalg.norm(U[:, 2])
    best, valid = None, np.zeros(0, int)
    for R, tt in ((R1, t), (R2, t), (R1, -t), (R2, -t)):
        v = np.nonzero(cheirality(R, tt, r1, r2))[0]
        if len(v) >= len(valid):
            best, valid = (R, tt), v
    return best[0], best[1], valid


def midpoint(R, t, r1, r2):
    r2c = r2 @ R
    c2 = -(R.T @ t)
    A = np.stack([r1, -r2c, np.broadcast_to(-c2, r1.shape)], 2)
    v = np.linalg.svd(A)[2][:, 2, :]
    ok = v[:, 2] != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = v[:, :2] / v[:, 2:3]
    ok &= (lam[:, 0] > np.finfo(float).eps) & (lam[:, 1] > np.finfo(float).eps)
    p = 0.5 * (lam[:, :1] * r1 + c2 + lam[:, 1:2] * r2c)
    return p[ok], ok


def opp_minor(m, row, col):
    c1, c2 = (1 if col == 0 else 0), (1 if col == 2 else 2)
    r1, r2 = (1 if row == 0 else 0), (1 if row == 2 else 2)
    return m[r1, c2] * m[r2, c1] - m[r1, c1] * m[r2, c2]


def decompose_homography(H, K):
    Hn = np.linalg.inv(K) @ H @ K
    U, s, Vt = np.linalg.svd(Hn)
    Hn = Hn / s[1]
    sign = 1.0
    if np.linalg.det(Hn) < 0:
        Hn, sign = -Hn, -1.0
    S = Hn.T @ Hn - np.eye(3)
    if np.abs(S).max() < 1e-3:
        return [(sign * U @ Vt, np.zeros(3))]
    rt = lambda x: np.sqrt(max(x, 0.0))
    rM00, rM11, rM22 = rt(opp_minor(S, 0, 0)), rt(opp_minor(S, 1, 1)), rt(opp_minor(S, 2, 2))
    e12, e02, e01 = np.sign(opp_minor(S, 1, 2)), np.sign(opp_minor(S, 0, 2)), np.sign(opp_minor(S, 0, 1))
    idx = int(np.argmax([abs(S[0, 0]), abs(S[1, 1]), abs(S[2, 2])]))
    n1, n2 = np.zeros(3), np.zeros(3)
    if idx == 0:
        n1[:] = [S[0, 0], S[0, 1] + rM22, S[0, 2] + e12 * rM11]
        n2[:] = [S[0, 0], S[0, 1] - rM22, S[0, 2] - e12 * rM11]
    elif idx == 1:
        n1[:] = [S[0, 1] + rM22, S[1, 1], S[1, 2] - e02 * rM00]
        n2[:] = [S[0, 1] - rM22, S[1, 1], S[1, 2] + e02 * rM00]
    else:
        n1[:] = [S[0, 2] + e01 * rM11, S[1, 2] + rM00, S[2, 2]]
        n2[:] = [S[0, 2] - e01 * rM11, S[1, 2] - rM00, S[2, 2]]
    v = 2.0 * (s[0] / s[1]) * (s[2] / s[1])
    r = rt(2 + np.trace(S) + v)
    n_t = rt(2 + np.trace(S) - v)
    n1, n2 = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2)
    es = np.sign(S[idx, idx])
    t1s = 0.5 * n_t * (es * r * n2 - n_t * n1)
    t2s = 0.5 * n_t * (es * r * n1 - n_t * n2)
    R1 = Hn @ (np.eye(3) - (2.0 / v) * np.outer(t1s, n1))
    R2 = Hn @ (np.eye(3) - (2.0 / v) * np.outer(t2s, n2))
    t1, t2 = R1 @ t1s, R2 @ t2s
    return [(R1, t1), (R1, -t1), (R2, t2), (R2, -t2)]


def pose_from_homography(H, K, r1, r2):
    R_b, t_b, pts_b, err_b = np.eye(3), np.zeros(3), np.zeros((0, 3)), np.inf
    for R, t in decompose_homography(H, K):
        p, ok = midpoint(R, t, r1, r2)
        err = 0.0
        if len(p):
            q = p @ R.T + t
            e1 = 1 - np.clip((r1[ok] * p).sum(1) / np.linalg.norm(p, axis=1), -1, 1)
            e2 = 1 - np.clip((r2[ok] * q).sum(1) / np.linalg.norm(q, axis=1), -1, 1)
            err = float((e1 + e2).sum())
        if len(p) > len(pts_b) or (len(p) == len(pts_b) and err < err_b):
            R_b, t_b, pts_b, err_b = R, t, p, err
    return R_b, t_b, pts_b


def label_of(cfg, angle):
    if cfg in (UNCALIBRATED, PLANAR) and angle is not None and angle < C.MIN_TRI_ANGLE:
        return "panoramic"
    return {UNCALIBRATED: "general", PLANAR: "planar", PANORAMIC: "panoramic"}.get(cfg, "rejected")


def two_view(p1, p2, r1, r2, K, rng):
    n = len(p1)
    g = {"config": DEGENERATE, "mask": None, "angle": None, "n_putative": n, "n_f": 0, "n_h": 0}
    if n < C.MIN_NUM_INLIERS:
        g["label"], g["reason"] = "rejected", f"{n} putative matches < {C.MIN_NUM_INLIERS}"
        return g
    F = loransac(p1, p2, SevenPoint(), EightPoint(), rng)
    ratio = max(C.MIN_INLIER_RATIO, C.MAX_H_INLIER_RATIO * F["inliers"] / n)
    H = loransac(p1, p2, Homography(True), Homography(), rng, min_inlier_ratio=ratio)
    g["n_f"], g["n_h"] = F["inliers"], H["inliers"]
    passed = lambda r: r["success"] and r["inliers"] >= C.MIN_NUM_INLIERS
    if not (passed(F) or passed(H)):
        g["label"], g["reason"] = "rejected", f"F {F['inliers']}, H {H['inliers']} inliers < {C.MIN_NUM_INLIERS}"
        return g
    h_f = H["inliers"] / F["inliers"] if F["inliers"] else np.inf
    g["h_f"] = None if np.isinf(h_f) else float(h_f)
    if h_f > C.MAX_H_INLIER_RATIO:
        cfg = PLANAR_OR_PANORAMIC
        use = H if H["inliers"] >= F["inliers"] else F
    else:
        cfg, use = UNCALIBRATED, F
    mask = use["mask"]
    idx = np.nonzero(mask)[0]
    angle = None
    if cfg == UNCALIBRATED:
        R, t, valid = pose_from_essential(K.T @ F["model"] @ K, r1[idx], r2[idx])
        if len(valid):
            a = fold(angle_between(r1[idx][valid], r2[idx][valid] @ R))
            angle = float(np.degrees(np.median(a)))
    else:
        R, t, pts = pose_from_homography(H["model"], K, r1[idx], r2[idx])
        if t @ t < C.PANORAMIC_T_SQ:
            cfg, angle = PANORAMIC, 0.0
        else:
            cfg = PLANAR
            if len(pts):
                c2 = -(R.T @ t)
                a = fold(angle_between(pts, pts - c2))
                angle = float(np.degrees(np.median(a)))
    g.update(config=cfg, mask=mask, angle=angle, label=label_of(cfg, angle))
    g["reason"] = (f"{cfg}: F {F['inliers']}, H {H['inliers']}, {int(mask.sum())} inliers"
                   + (f", {angle:.1f} deg" if angle is not None else ""))
    return g
