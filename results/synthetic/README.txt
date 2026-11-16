================================================================================
SYNTHETIC TEST SCENE - how to use it
================================================================================
A fake scene where the correct answer is known. Build and test Stage 3 onward
against it before any real photo is matched. Stage 2 is not needed - the
descriptors are already provided.

--------------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------------
  results/synthetic/clean.npz  noisy.npz  outliers.npz    the data
  scripts/synthetic/load_synthetic.py                     import this
  scripts/synthetic/selftest.py                           checks the data is sane
  scripts/synthetic/make_synthetic.py                     regenerates the data

You only need load_synthetic.py. It finds the .npz files on its own.

--------------------------------------------------------------------------------
LOADING
--------------------------------------------------------------------------------
    import sys; sys.path.insert(0, "scripts/synthetic")
    from load_synthetic import load

    S = load("clean")          # or "noisy" or "outliers"

--------------------------------------------------------------------------------
YOUR INPUT
--------------------------------------------------------------------------------
    S.K                  (3,3)    shared intrinsics, never refined
    S.dist               (5,)     all zeros - images are already undistorted
    S.calib_resolution   (3024, 4032)
    S.n_cameras          37

    S.features[i]     -> (kpts, desc, colors)
                           kpts   (N,2) float32    pixel coordinates
                           desc   (N,128) float32  descriptors
                           colors (N,3) uint8

    S.matches[(i,j)]  -> (M,2) int32
                           Keypoint INDICES, not coordinates. Row [a, b] means
                           keypoint a of image i matches keypoint b of image j.

    S.pairs()             every pair that has matches
    S.pairs("general")    only pairs with that ground-truth label

--------------------------------------------------------------------------------
GROUND TRUTH  (to check your output against - a real pipeline never sees this)
--------------------------------------------------------------------------------
    S.gt_poses[i]        -> (rvec, tvec)    X_cam = R @ X_world + tvec
    S.gt_centres[i]      -> (3,)            camera position in the world
    S.gt_xyz             (P,3)              the true 3D points
    S.gt_rgb             (P,3) uint8

    S.gt_kp_point_id[i]  -> (N,) int32
          Which 3D point each keypoint came from; -1 means none.
          A match [a, b] is CORRECT when
              S.gt_kp_point_id[i][a] == S.gt_kp_point_id[j][b]

    S.gt_pair_label[(i,j)]   -> 'general' | 'planar' | 'panoramic' | 'rejected'
    S.gt_pair_alpha[(i,j)]   -> median triangulation angle, degrees
    S.gt_match_inlier[(i,j)] -> (M,) bool, True where that match row is correct
                                (all True except in the "outliers" variant)

--------------------------------------------------------------------------------
THE THREE VARIANTS - use them in this order
--------------------------------------------------------------------------------
  clean      No noise, every match correct. Your results should come out almost
             exactly right. If they do not, it is a bug - do not blame noise.

  noisy      1.0 px Gaussian noise on every keypoint; matches still all correct.
             Use to tune thresholds and check convergence. Bundle adjustment
             should settle near 1.41 px and stop - lower means fitting noise.

  outliers   Same noise, plus 30% of the rows in every matches[(i,j)] are
             deliberately wrong. Use to test RANSAC; gt_match_inlier says
             exactly which rows are wrong, so you can measure what you caught.

  Only the MATCHES are corrupted in "outliers". Descriptors stay correct in all
  three variants.

--------------------------------------------------------------------------------
THE SCENE
--------------------------------------------------------------------------------
  37 cameras, 8600 points, 232 pairs with matches.

    cameras 00-29    a full 360 deg circle around an object, 12 deg apart
    cameras 30-32    same position as camera 00, only rotated
    cameras 33-36    pointed at a flat wall

  That gives all four pair labels, so every branch of your Stage 3b decision
  tree gets tested:

    general   149    normal pairs; these may seed
    planar     45    the flat wall; must NOT seed
    panoramic   6    zero baseline (cameras 30-32); must NOT seed or triangulate
    rejected   32    too few matches

  The 6 panoramic pairs are the usual failure - they must not come out planar.

--------------------------------------------------------------------------------
FOUR CONVENTIONS THAT WILL BITE YOU
--------------------------------------------------------------------------------
  1. tvec is the TRANSLATION, not the camera position.
     X_cam = R @ X_world + tvec.    Camera position = -R.T @ tvec.

  2. dist is all zeros. The images are already undistorted. Do not apply
     distortion anywhere.

  3. matches hold integer INDICES, never pixel coordinates.

  4. Thresholds are in PIXELS. Fitting E on calibrated rays needs the pixel
     threshold divided by f = (K[0,0] + K[1,1]) / 2. Forget this and you are
     wrong by a factor of ~3000, but the pipeline still runs, so you will not
     notice.

--------------------------------------------------------------------------------
REGENERATING  (only for a different seed or size)
--------------------------------------------------------------------------------
    python scripts/synthetic/make_synthetic.py --seed 1234
    python scripts/synthetic/selftest.py
================================================================================
