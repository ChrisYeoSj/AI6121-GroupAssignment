import json
import os
import cv2
import numpy as np
import config as C

SQ_NORM = 512.0 * 512.0


def image_names(scene):
    d = os.path.join(C.PHOTOS, scene)
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg")))


def feature_path(scene, name):
    return os.path.join(C.OUT, scene, "features", os.path.splitext(name)[0] + ".npz")


def cache_key(path):
    s = os.stat(path)
    return (f"{s.st_size}:{s.st_mtime_ns}:{C.MAX_NUM_FEATURES}:{C.MAX_IMAGE_SIZE}:"
            f"{C.PEAK_THRESHOLD}:{C.EDGE_THRESHOLD}:{C.CACHE_VERSION}")


def detect(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(path)
    h, w = bgr.shape[:2]
    scale = max(w, h) / C.MAX_IMAGE_SIZE
    sw, sh = (round(w / scale), round(h / scale)) if scale > 1 else (w, h)
    gray = cv2.cvtColor(cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA) if scale > 1 else bgr,
                        cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=C.PEAK_THRESHOLD * 3, edgeThreshold=C.EDGE_THRESHOLD)
    kp, desc = sift.detectAndCompute(gray, None)
    order = np.argsort([-k.size for k in kp], kind="stable")[:C.MAX_NUM_FEATURES]
    xy = np.array([kp[i].pt for i in order], np.float32)
    xy = np.c_[(xy[:, 0] + 0.5) * (w / sw) - 0.5, (xy[:, 1] + 0.5) * (h / sh) - 0.5].astype(np.float32)
    raw = np.clip(np.rint(desc[order]), 0, 255).astype(np.uint8)
    return dict(kpts=xy, desc_raw=raw, wh=np.array([w, h]))


def normalise(raw, norm):
    d = raw.astype(np.float32)
    if norm == "L1_ROOT":
        n = d.sum(1, keepdims=True)
        d = np.sqrt(d / np.where(n > 0, n, 1))
    else:
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(n > 0, n, 1)
    return np.clip(np.rint(512 * d), 0, 255).astype(np.uint8)


def extract_scene(scene):
    os.makedirs(os.path.join(C.OUT, scene, "features"), exist_ok=True)
    names = image_names(scene)
    done = 0
    for nm in names:
        src, dst = os.path.join(C.PHOTOS, scene, nm), feature_path(scene, nm)
        key = cache_key(src)
        if os.path.isfile(dst):
            with np.load(dst) as z:
                if str(z["key"]) == key:
                    continue
        np.savez_compressed(dst, key=key, **detect(src))
        done += 1
    print(f"  features: {len(names)} images, {len(names) - done} cached, {done} computed", flush=True)
    return names


def load_calibration(scene):
    with np.load(feature_path(scene, image_names(scene)[0])) as z:
        w, h = (int(v) for v in z["wh"])
    with open(C.CALIB) as f:
        c = json.load(f)
    K = np.array(c["K"], np.float64).reshape(3, 3)
    dist = np.array(c.get("dist", [0] * 5), np.float64).ravel()
    cw, ch = c.get("resolution") or c.get("calib_resolution")
    if (cw, ch) != (w, h):
        sx, sy = w / cw, h / ch
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return K, dist


def load_image_features(scene, name, norm):
    with np.load(feature_path(scene, name)) as z:
        kpts, raw = z["kpts"].astype(np.float64), z["desc_raw"]
    return kpts, normalise(raw, norm).astype(np.float32)


def one_way(dots, ratio, max_distance, axis):
    idx = dots.argmax(axis)
    b = np.expand_dims(idx, axis)
    best = np.take_along_axis(dots, b, axis)
    np.put_along_axis(dots, b, 0, axis)
    second = dots.max(axis, keepdims=True)
    np.put_along_axis(dots, b, best, axis)
    best, second = best.ravel(), second.ravel()
    d1 = np.arccos(np.minimum(best / SQ_NORM, 1.0))
    d2 = np.arccos(np.minimum(second / SQ_NORM, 1.0))
    keep = (best > 0) & (d1 <= max_distance) & (d1 < ratio * d2)
    return np.where(keep, idx, -1)


def match(d1, d2, ratio=C.MAX_RATIO, max_distance=C.MAX_DISTANCE):
    dots = d1 @ d2.T
    m12 = one_way(dots, ratio, max_distance, 1)
    m21 = one_way(dots, ratio, max_distance, 0)
    i = np.nonzero(m12 >= 0)[0]
    i = i[m21[m12[i]] == i]
    return np.c_[i, m12[i]].astype(np.int32)
