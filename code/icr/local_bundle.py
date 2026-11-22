"""Choose a local bundle using shared reconstructed points."""
import numpy as np


def select_local_bundle(camera_ids, cam_idx, point_idx, newest_camera, window_size=6):
    """Keep all observations of local points, including those in fixed cameras."""
    if window_size < 2:
        raise ValueError("A local window needs at least two cameras.")
    newest = camera_ids.index(newest_camera)
    newest_points = np.unique(point_idx[cam_idx == newest])
    shared = {}
    for row in range(len(camera_ids)):
        if row != newest:
            count = np.intersect1d(newest_points, point_idx[cam_idx == row]).size
            if count:
                shared[row] = int(count)
    # Prefer more recent registrations when shared-point counts are equal.
    neighbours = sorted(shared, key=lambda row: (shared[row], row), reverse=True)
    rows = [newest] + neighbours[:window_size - 1]
    points = np.unique(point_idx[np.isin(cam_idx, rows)])
    mask = np.isin(point_idx, points)
    observed_rows = np.unique(cam_idx[mask])
    # Seed cameras anchor the reconstruction even if selected in the local window.
    variable_rows = [row for row in rows if row >= 2]
    return dict(camera_ids=[camera_ids[row] for row in rows],
                variable_camera_ids=[camera_ids[row] for row in variable_rows],
                point_indices=points, observation_mask=mask,
                fixed_camera_ids=[camera_ids[row] for row in observed_rows
                                  if row not in variable_rows],
                shared_points={int(camera_ids[row]): shared[row] for row in neighbours})
