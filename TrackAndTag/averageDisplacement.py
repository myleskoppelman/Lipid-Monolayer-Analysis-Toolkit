import numpy as np
import pandas as pd
from tqdm import tqdm

''' [averaveDisplacement.py] Last Updated: 5/30/2025 by Myles Koppelman '''



def average(particles, n_frames):
    '''
    [averageDisplacement.py] Last Updated: 5/30/2025 by Myles Koppelman 
    
    
    Calculate average frame-to-frame displacements in x and y directions for tracked particles,
    to be used in iterative tracking.

    For each particle trajectory (represented as a numpy array of observations),
    this function computes the displacement vectors between consecutive frames,
    filters out outliers beyond 2 standard deviations from the mean displacement,
    and calculates the average displacement per frame across all particles.

    Args:
        particles (list of np.ndarray): Each element is a 2D array for a single particle,
            where columns correspond to properties including at least:
            - column 1: x position,
            - column 2: y position,
            - column 3: frame index.
        n_frames (int): Total number of frames in the dataset.

    Returns:
        tuple of np.ndarray: (dx, dy), each of length n_frames,
            where dx[i] and dy[i] represent the average displacement in the x and y
            directions, respectively, from frame i to frame i+1.
            Frames with no displacement data return zero.

    Notes:
        - Outliers in displacement are filtered using a 2-standard-deviation cutoff.
        - Displacements are computed as differences in positions between consecutive frames.
        - The function handles cases where no particles move between frames by assigning zero displacement.
    '''
    
    data = []

    for particle in tqdm(particles, desc='Calculating Displacements...'):
        frame = particle[1:, 3]
        xdisp = particle[1:, 1] - particle[:-1, 1]
        ydisp = particle[1:, 2] - particle[:-1, 2]
        data.append(np.column_stack((frame, xdisp, ydisp)))

    displacements = np.vstack(data)
    dx = np.zeros(n_frames)
    dy = np.zeros(n_frames)

    for n in tqdm(range(1, n_frames + 1), desc='Calculating Averages...'):
        rows = displacements[:, 0] == n
        x_vals = displacements[rows, 1]
        y_vals = displacements[rows, 2]

        if len(x_vals) == 0 or len(y_vals) == 0:
            dx[n-1] = 0
            dy[n-1] = 0
            continue

        def filter_outliers(values):
            mean_val = np.mean(values)
            std_val = np.std(values)
            filtered = values[np.abs(values - mean_val) <= 2 * std_val]
            return filtered

        x_filtered = filter_outliers(x_vals)
        y_filtered = filter_outliers(y_vals)

        dx[n-1] = np.mean(x_filtered) if x_filtered.size > 0 else 0
        dy[n-1] = np.mean(y_filtered) if y_filtered.size > 0 else 0

    return dx, dy
