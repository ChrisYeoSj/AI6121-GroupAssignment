# Optimisation experiments

The existing global BA solver is extended with a local window and two schedules.
`code/icr/local_bundle.py` selects up to six cameras by shared points;
`entrypoint.py` selects the schedule. The first two camera poses stay fixed.

Use Python 3.12 and run from the repository root:

```sh
python -m pip install -r requirements-evaluation.txt
python scripts/evaluation/run_baseline.py --scene yunnan_rock_1 --mode final_only --out results/evaluation/rock1_final
python scripts/evaluation/run_baseline.py --scene yunnan_rock_1 --mode local_final --out results/evaluation/rock1_local
```

Use a fresh output folder for each run. Replace the scene with
`yunnan_rock_1_lower_height`, `yunnan_rock_2` or `yunnan_rock_2_lower_height`
for the other datasets. The runners read the existing calibration and cached
RootSIFT features, matches and tracks under `results`; download their Git LFS
files if needed. Feature extraction is not repeated.

`metrics.json` contains counts, errors and runtime. `ba_steps.json` records the
solver status and errors before and after each BA call. Status 0 means the
evaluation limit was reached. To repeat the local budget check, add
`--local-max-nfev 300`; the default local and global limits are 100.

For synthetic inputs:

```sh
python scripts/evaluation/run_synthetic_pipeline.py --data results/synthetic --out results/evaluation/synthetic
```

The input directory contains `clean.npz`, `noisy.npz` and `outliers.npz`.
This test uses supplied keypoints and matches, then compares recovered camera
centres with ground truth after alignment. Exact report numbers require the
archived input files used for those runs; newly generated noise can differ.

Local verification and comparison helpers are not part of this minimal submission.
The original solver attribution remains in `code/icr/bundle_adjustment.py`.
