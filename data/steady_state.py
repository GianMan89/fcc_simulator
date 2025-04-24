"""
steady_state.py

Provides steady-state initial conditions for FCC–Fractionator simulation.
Simply import this module to access the variables.
"""
import os
from scipy.io import loadmat
import numpy as np

# Locate the .mat file in the same directory
_mat_path = os.path.join(os.path.dirname(__file__), 'steady_state.mat')
_ss = loadmat(_mat_path)

# Reactor states (1×37)
xfcc = _ss['xfcc'].flatten()

# FCC manipulated variables (1×15)
ufcc = _ss['ufcc'].flatten()

# Disturbances (should be length 9 or more)
dist = _ss.get('dist', None)
if dist is not None:
    dist = dist.flatten()

# LPG flow (scalar)
Flpg = float(_ss['Flpg'])

# Condenser temperature (scalar)
Tcondenser = float(_ss['Tcondenser'])

# Fractionator manipulated variables (e.g. duties/flows) [6×1]
MV = _ss['MV'].flatten()

# Fractionator setpoints [4×1]
SP = _ss['SP'].flatten()

# Product flows [3×1]
products = _ss['products'].flatten()

# Integral controller errors for fractionator [4×1]
errord = _ss['errord'].flatten()

# Filter initial state [13×1]
Xfilin = _ss['Xfilin'].flatten()

# Liquid composition matrix [20×10]
xc = np.array(_ss['xc'], dtype=float)

# Fractionator dynamic states [80×1]
xfra = _ss['xfra'].flatten()

# Fractionator manipulated variables (for dufra) [40×1]
ufra = _ss['ufra'].flatten()

# Optional: Distillate initial
Distillateini = float(_ss.get('Distillateini', np.nan))

# Clean up namespace
del _ss, _mat_path, loadmat
