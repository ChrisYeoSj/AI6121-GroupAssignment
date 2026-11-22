"""Evaluate existing geometry/mapping from synthetic keypoints, without GT inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import sys, json, time, traceback, hashlib
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
for path in ['scripts/synthetic','scripts/geometry','scripts/correspondence_search','code/icr']:
    sys.path.insert(0,str(ROOT/path))
from load_synthetic import load
from utils.pipeline import build_tracks
from utils.verification import two_view
from two_view_opencv import initialise_reconstruction
import entrypoint as inc
from run_baseline import save_model, write_json


def main():
    parser = argparse.ArgumentParser(description='Evaluate synthetic keypoints and matches.')
    parser.add_argument('--data', type=Path, default=ROOT/'results/synthetic')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=['clean', 'noisy', 'outliers'],
                        default=['clean', 'noisy', 'outliers'])
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        if not (args.data/f'{variant}.npz').is_file():
            parser.error('Missing synthetic input: ' + str(args.data/f'{variant}.npz'))
    cv2.setNumThreads(1)
    failed = False
    for variant in args.variants:
        folder=OUT/variant
        folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():
            raise FileExistsError('Use a fresh output directory: ' + str(folder))
        began=time.perf_counter()
        status=dict(state='running',variant=variant,seed=0,mode='final_only',
                    scope='synthetic keypoints and putative matches through verification, tracks, initialisation, mapping and final BA; no image feature extraction')
        def progress(stage):
            status.update(stage=stage,elapsed_seconds=time.perf_counter()-began)
            write_json(folder/'status.json',status)
            print(variant,stage,flush=True)
        progress('loading')
        try:
            with (folder/'run.log').open('w',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
                path=args.data/f'{variant}.npz'
                s=load(variant,path=path)
                status['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                kpts=[s.features[i][0].astype(float) for i in range(s.n_cameras)]
                usable={}; labels={}
                cv2.setRNGSeed(0)
                for count,((i,j),m) in enumerate(sorted(s.matches.items()),1):
                    g=two_view(kpts[i][m[:,0]],kpts[j][m[:,1]],s.K,np.random.default_rng(i*100003+j))
                    labels[str((i,j))]=g['label']
                    if g['label']!='rejected' and g['mask'] is not None:
                        usable[i,j]=m[g['mask']]
                    if count%20==0: progress(f'verified {count}/{len(s.matches)} pairs')
                obs,lens,conflicts=build_tracks([len(x) for x in kpts],usable)
                tracks={i:[] for i in range(len(lens))}
                for tid,cam,kp in obs: tracks[int(tid)].append((int(cam),int(kp)))
                status.update(tracks=len(tracks),conflicted_tracks=int(conflicts),usable_pairs=len(usable))
                np.savez_compressed(folder/'tracks.npz',observations=obs,track_lengths=lens)
                write_json(folder/'pair_labels.json',labels)
                progress('initialising')
                start=initialise_reconstruction(s.K,kpts,usable,0)
                status['seed_pair']=[int(start.seed.first_image),int(start.seed.second_image)]
                data=dict(K=s.K,kpts=kpts,matches=usable,tracks=tracks)
                progress('incremental reconstruction and final BA')
                poses,points,tp=inc.entrypoint(data,start,ba_mode='final_only')
                metrics=save_model(folder,'final_model',poses,points,dict(features=tracks,points=tp),kpts,s.K,inc)
                metrics.update(input_cameras=s.n_cameras,missing_camera_ids=sorted(set(range(s.n_cameras))-set(poses)))
                # Ground truth is used only after reconstruction for camera-centre evaluation.
                ids=list(poses)
                a=np.array([-poses[i][0].T@np.ravel(poses[i][1]) for i in ids])
                b=np.array([s.gt_centres[i] for i in ids])
                ac=a-a.mean(0);bc=b-b.mean(0)
                U,D,Vt=np.linalg.svd(bc.T@ac/len(a))
                sign=np.eye(3);sign[-1,-1]=np.linalg.det(U@Vt)
                Q=U@sign@Vt
                scale=float(np.sum(D*np.diag(sign))/np.mean(np.sum(ac*ac,axis=1)))
                aligned=scale*ac@Q.T+b.mean(0)
                metrics['aligned_camera_centre_rmse_gt_units']=float(np.sqrt(np.mean(np.sum((aligned-b)**2,axis=1))))
                metrics['alignment_cameras']=len(ids)
                write_json(folder/'metrics.json',metrics)
                status.update(state='completed',registered_cameras=len(poses),input_cameras=s.n_cameras)
                progress('completed')
        except Exception as e:
            failed = True
            status.update(state='failed',error=str(e),traceback=traceback.format_exc())
            progress('failed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
