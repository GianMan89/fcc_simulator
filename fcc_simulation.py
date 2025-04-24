"""
fcc_simulation.py

Main script to run the unified FCC–Fractionator simulation.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

from data.steady_state import xfcc, ufcc, dist, Flpg, Tcondenser, MV, SP, products, errord, Xfilin, xc, xfra, ufra
from fcc.dynamics import fcc_dynamics
from fractionator.dynamics import fractionator_dynamics
from utils.disturbances import interp_disturbance
from utils.filters import filter_derivative


def load_params():
    """
    Assemble simulation parameters from steady-state and defaults.
    """
    params = {}
    params['initial'] = {
        'ufcc':        ufcc.copy(),
        'dist':        dist.copy(),
        'Flpg':        float(Flpg),
        'Tcondenser':  float(Tcondenser),
        'MV':          MV.copy(),
        'SP':          SP.copy(),
        'products':    products.copy(),
        'errord':      errord.copy(),
        'Xfilin':      Xfilin.copy(),
        'xc':          xc.copy(),
        'xfra':        xfra.copy(),
        'ufra':        ufra.copy()
    }
    # FCC controller settings
    params['fccVariance'] = 1.0
    params['fcc_e_old'] = np.zeros(6)
    params['FCC_PID'] = {
        'Kc':  np.array([1500.0, 1.5, 1500.0, 0.0001, 5.0, 0.5]),
        'Tao': np.array([200.0, 800.0, 200.0, 1000.0, 100.0, 200.0])
    }
    # Simulation settings
    params['dt'] = 10.0             # PID & filter time step (s)
    params['filterTC'] = 1800.0     # Filter time constant (s)
    # Fractionator settings
    params['Nstages'] = xc.shape[0]
    params['flashDelta'] = 10.0/3600.0  # hours per stage step
    params['valve_min_pct'] = 10
    params['valve_max_pct'] = 150
    # Dummy stage profiles (override as needed)
    params['P_profile'] = np.ones(params['Nstages']) * xfra[4]  # use P5 initial as proxy
    params['T_profile'] = np.ones(params['Nstages']) * xfcc[1]  # use T2 initial as proxy
    # PID settings for fractionator
    params['L1_PID']  = (1.0, 60.0)   # (K, Ti)
    params['TC_PID'] = (1.0, 120.0)
    params['HN_PID'] = (1.0, 90.0)
    params['LCO_PID']= (1.0, 90.0)
    return params


def combined_dynamics(t, x, params, disturbance_profile):
    """
    Combined ODE function for reactor, filter, and fractionator.
    """
    # Determine state partition sizes from steady-state definitions
    nR   = xfcc.shape[0]    # number of FCC states
    nF   = Xfilin.shape[0]  # number of filter states
    nFra = xfra.shape[0]    # number of fractionator states
    nU   = ufra.shape[0]    # number of fractionator MVs

    # Unpack state vector
    idx = 0
    xfcc_st = x[idx:idx+nR]; idx += nR
    xfil_st = x[idx:idx+nF]; idx += nF
    xfra_st = x[idx:idx+nFra]; idx += nFra
    ufra_st = x[idx:idx+nU]

    # Interpolate disturbances
    dist_now = interp_disturbance(t, disturbance_profile, params['initial']['dist'])

    # FCC reactor dynamics
    dxfcc, yp, fcc_e_new = fcc_dynamics(xfcc_st, dist_now, params)
    params['fcc_e_old'] = fcc_e_new

    # Filter dynamics (first-order)
    dxfil = filter_derivative(xfil_st, yp[:nF], params['filterTC'])

    # Fractionator dynamics
    dxfra, dufra = fractionator_dynamics(xfra_st, ufra_st, xfil_st, params)

    # Pack derivatives
    return np.concatenate([dxfcc, dxfil, dxfra, dufra])([dxfcc, dxfil, dxfra, dufra])


def dynamic_driver(disturbance_profile=None, sim_minutes=60):
    """
    Run the full FCC–Fractionator simulation.

    Args:
        disturbance_profile (list of dict): [{'time': s, 'values': array}, ...]
        sim_minutes (float): Total simulation time in minutes.

    Returns:
        dict: results with keys 'time', 'xfcc', 'xfil', 'xfra', 'ufra'.
    """
    # Determine state partition sizes from steady-state definitions
    nR   = xfcc.shape[0]    # number of FCC states
    nF   = Xfilin.shape[0]  # number of filter states
    nFra = xfra.shape[0]    # number of fractionator states
    nU   = ufra.shape[0]    # number of fractionator MVs
    
    if disturbance_profile is None:
        disturbance_profile = []
    params = load_params()
    # Initial state
    x0 = np.concatenate([xfcc, Xfilin, xfra, ufra])
    t_end = sim_minutes * 60.0
    t_eval = np.arange(0, t_end+params['dt'], params['dt'])

    sol = solve_ivp(lambda t, x: combined_dynamics(t, x, params, disturbance_profile),
                    (0, t_end), x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)

    # Unpack results
    res = {}
    res['time'] = sol.t
    y = sol.y.T
    idx = 0
    res['xfcc'] = y[:, idx:idx+nR]; idx+=nR
    res['xfil'] = y[:, idx:idx+nF]; idx+=nF
    res['xfra'] = y[:, idx:idx+nFra]; idx+=nFra
    res['ufra'] = y[:, idx:idx+nU]
    return res


if __name__ == '__main__':
    # Example run: no disturbances, 10 minutes
    results = dynamic_driver([], sim_minutes=10)
    print('Simulation complete.')
    print('Time points:', results['time'].shape)
