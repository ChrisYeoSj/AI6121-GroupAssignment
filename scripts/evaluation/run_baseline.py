"""Run the integrated pipeline and retain the model, observations and measurements."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/geometry"))
import initialise_from_features as pipeline
import two_view_opencv as geometry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def measure(poses, points, ci, pi, pixels, K):
    cameras = list(poses)
    predicted = np.empty_like(pixels, dtype=float)
    depths = np.empty(len(pixels))
    for row, camera in enumerate(cameras):
        mask = ci == row
        R, t = poses[camera]
        cam_points = points[pi[mask]] @ R.T + np.ravel(t)
        projected = cam_points @ K.T
        predicted[mask] = projected[:, :2] / projected[:, 2:]
        depths[mask] = cam_points[:, 2]
    errors = np.linalg.norm(predicted - pixels, axis=1)
    finite = np.isfinite(errors)
    selected = errors[finite]
    return dict(observations=len(errors), nonfinite_errors=int((~finite).sum()),
                behind_camera=int((depths <= 0).sum()),
                mean_error_px=float(selected.mean()) if len(selected) else None,
                median_error_px=float(np.median(selected)) if len(selected) else None,
                rmse_px=float(np.sqrt(np.mean(selected**2))) if len(selected) else None,
                max_error_px=float(selected.max()) if len(selected) else None)


def save_model(folder, prefix, poses, points, tracks, keypoints, K, module):
    ci, pi, pixels, camera_ids = module.collect_observations(
        poses, tracks["features"], tracks["points"], keypoints)
    used = np.unique(pi)
    pairs = [(track, point) for track, ids in tracks["points"].items() for point in ids]
    np.savez_compressed(folder/(prefix+".npz"), K=K, points=points,
                        camera_ids=np.array(camera_ids),
                        rotation_matrices=np.array([poses[i][0] for i in camera_ids]),
                        translations=np.array([np.ravel(poses[i][1]) for i in camera_ids]),
                        observation_camera_ids=np.array(camera_ids)[ci],
                        observation_point_indices=pi, observation_pixels=pixels,
                        track_point_map=np.array(pairs,dtype=np.int64).reshape(-1,2))
    metrics = measure(poses,points,ci,pi,pixels,K)
    metrics.update(total_points=len(points), observed_points=len(used),
                   unobserved_points=len(points)-len(used), registered_cameras=len(poses))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="yunnan_rock_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("global_each", "local_final", "final_only"), default="global_each")
    p.add_argument("--local-window", type=int, default=6)
    p.add_argument("--local-max-nfev", type=int, default=100)
    p.add_argument("--global-max-nfev", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT/"results/evaluation/baseline_20261121")
    args = p.parse_args()
    if min(args.local_max_nfev, args.global_max_nfev) < 1:
        p.error("BA evaluation limits must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"status.json").exists():
        raise FileExistsError("Use a fresh --out directory to preserve previous results.")
    status = dict(state="running",scene=args.scene,seed=args.seed,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  reference_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  mode=args.mode, local_window=args.local_window,
                  local_max_nfev=args.local_max_nfev, global_max_nfev=args.global_max_nfev,
                  filtering="same depth/error/angle filter after every registration and after final BA")
    write_json(out/"status.json",status)
    paths = [ROOT/"results/calibration/calibration.json"]
    paths += sorted((ROOT/"results/correspondence_search"/args.scene).rglob("*"))
    paths += sorted((ROOT/"scripts").rglob("*.py"))+sorted((ROOT/"code").rglob("*.py"))
    hashes = {str(f.relative_to(ROOT)):digest(f) for f in paths if f.is_file()}
    write_json(out/"inputs_and_code.json",hashes)
    write_json(out/"environment.json",dict(python=sys.version,system=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,
                    opencv_threads=1,blas_threads=1))
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    original_stdout = sys.stdout
    began = time.perf_counter()
    steps = []
    def progress(text):
        print(text, file=original_stdout, flush=True)
        status.update(elapsed_seconds=time.perf_counter()-began,progress=text)
        write_json(out/"status.json",status)
    try:
        with (out/"run.log").open("w",encoding="utf-8",buffering=1) as log:
            with redirect_stdout(log),redirect_stderr(log):
                progress("Loading features and calibration")
                data=pipeline.load_for_incremental(args.scene)
                names=data["image_names"]
                write_json(out/"input_summary.json",dict(images=names,feature_counts=[len(x) for x in data["kpts"]],
                           pairs=len(data["matches"]),tracks=len(data["tracks"]),K=np.asarray(data["K"]).tolist()))
                original_pair=geometry.analyse_one_pair_of_image
                count=0
                def tracked_pair(*a,**kw):
                    nonlocal count
                    r=original_pair(*a,**kw)
                    count+=1
                    if count%100==0: progress(f"Initial geometry: {count}/{len(data['matches'])} pairs")
                    return r
                geometry.analyse_one_pair_of_image=tracked_pair
                start=geometry.initialise_reconstruction(data["K"],data["kpts"],data["matches"],args.seed)
                geometry.analyse_one_pair_of_image=original_pair
                status["initialisation_seconds"]=time.perf_counter()-began
                seed=[start.seed.first_image,start.seed.second_image]
                status["seed_pair"]=seed
                status["seed_names"]=[names[i] for i in seed]
                status["seed_points"]=len(start.points)
                progress(f"Seed: {status['seed_names']}, {len(start.points)} points")
                sys.path.insert(0,str(ROOT/"code/icr"))
                import entrypoint as inc
                track_points=inc.cr_track_points(data["tracks"],start.observations)
                initial_metrics=save_model(out,"seed_model",start.poses,start.points,
                            dict(features=data["tracks"],points=track_points),data["kpts"],data["K"],inc)
                write_json(out/"seed_metrics.json",initial_metrics)
                original_ba=inc.lsq
                def measured_ba(poses,points,ci,pi,pixels,K,**kwargs):
                    before=measure(poses,points,ci,pi,pixels,K)
                    kind="local" if "variable_camera_ids" in kwargs else "global"
                    progress(f"{kind.title()} BA: {len(poses)} cameras, {len(points)} points, {len(ci)} observations")
                    t=time.perf_counter()
                    limit=args.local_max_nfev if kind=="local" else args.global_max_nfev
                    fitted=original_ba(poses,points,ci,pi,pixels,K,return_info=True,max_nfev=limit,**kwargs)
                    seconds=time.perf_counter()-t
                    after=measure(fitted[0],fitted[1],ci,pi,pixels,K)
                    fixed=[int(list(poses)[i]) for i in np.unique(ci)
                           if list(poses)[i] not in fitted[2]["variable_cameras"]]
                    steps.append(dict(kind=kind,cameras=len(poses),points=len(points),seconds=seconds,
                                      before=before,after=after,solver=fitted[2],fixed_observing_cameras=fixed))
                    write_json(out/"ba_steps.json",steps)
                    np.savez_compressed(out/"latest_ba_checkpoint.npz",K=K,points=fitted[1],
                        camera_ids=np.array(list(fitted[0])),rotation_matrices=np.array([v[0] for v in fitted[0].values()]),
                        translations=np.array([np.ravel(v[1]) for v in fitted[0].values()]),
                        observation_camera_ids=np.array(list(fitted[0]))[ci],observation_point_indices=pi,observation_pixels=pixels)
                    progress(f"BA done: mean pixel error {before['mean_error_px']:.3f} -> {after['mean_error_px']:.3f}")
                    return fitted[:2]
                inc.lsq=measured_ba
                def before_final(poses,points,mapping):
                    metrics=save_model(out,"before_final_model",poses,points,
                        dict(features=data["tracks"],points=mapping),data["kpts"],data["K"],inc)
                    write_json(out/"before_final_metrics.json",metrics)
                poses,points,mapping=inc.entrypoint(data,start,ba_mode=args.mode,
                    local_window=args.local_window,before_final=before_final)
                final=save_model(out,"final_model",poses,points,dict(features=data["tracks"],points=mapping),
                                 data["kpts"],data["K"],inc)
                used=set()
                for ids in mapping.values(): used.update(ids)
                visible=points[sorted(used)]
                with (out/"point_cloud.ply").open("w",encoding="ascii") as f:
                    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(visible)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                    np.savetxt(f,visible,fmt="%.9g")
                final.update(input_images=len(names),registered_names=[names[i] for i in poses],
                             missing_names=[nm for i,nm in enumerate(names) if i not in poses],
                             mode=args.mode,global_ba_calls=sum(x["kind"]=="global" for x in steps),
                             local_max_nfev=args.local_max_nfev,global_max_nfev=args.global_max_nfev,
                             local_ba_calls=sum(x["kind"]=="local" for x in steps),
                             solver_limit_calls=sum(x["solver"]["status"]==0 for x in steps),
                             ba_seconds=sum(x["seconds"] for x in steps),
                             elapsed_seconds=time.perf_counter()-began,
                             mean_track_length=final["observations"]/max(1,final["observed_points"]))
                write_json(out/"metrics.json",final)
                print(json.dumps(final,indent=2))
                status["state"]="completed"
                progress(f"Completed: {len(poses)}/{len(names)} cameras, {final['observed_points']} observed points, mean error {final['mean_error_px']:.3f} px")
    except BaseException as exc:
        status.update(state="failed",error=repr(exc),elapsed_seconds=time.perf_counter()-began)
        with (out/"run.log").open("a",encoding="utf-8") as log:traceback.print_exc(file=log)
        write_json(out/"status.json",status)
        raise


if __name__=="__main__":
    main()
