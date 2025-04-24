"""
utils/disturbances.py

Provides functions to handle time‐based disturbance profiles.
"""
import numpy as np

def interp_disturbance(t, profile, base_dist):
    """
    Return the disturbance vector at time t.

    Args:
        t (float): Current time in seconds.
        profile (list of dict): Each dict has keys 'time' (float, seconds) and 'values' (ndarray).
        base_dist (ndarray): Default disturbance vector if no profile entry applies.

    Returns:
        ndarray: Disturbance vector for time t.
    """
    dist_now = base_dist.copy()
    if profile:
        # Extract times and ensure sorted order
        times = np.array([p['time'] for p in profile], dtype=float)
        idx = np.searchsorted(times, t, side='right') - 1
        if idx >= 0:
            dist_now = np.asarray(profile[idx]['values'], dtype=float)
    return dist_now
