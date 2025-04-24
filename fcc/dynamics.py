"""
fcc/dynamics.py

Full FCC reactor dynamics ported from MATLAB FCC.m.
Uses PID controllers, PFR kinetics, and enthalpy modules.
"""
import numpy as np
from utils.controllers import PIDController
from riser.pfr import PFR
from thermodynamics.enthalpy import enthalpy, enthalpyB


def fcc_dynamics(xfcc, dist, params):
    """
    Compute FCC reactor state derivatives and outputs.

    Args:
        xfcc (ndarray): Reactor state vector (37,).
        dist (ndarray): Disturbance vector (>=9,).
        params (dict): Parameter dict with:
            initial.ufcc, initial.Flpg, initial.Tcondenser,
            fccVariance, fcc_e_old (6,), FCC_PID dict, dt (sec), Mwin, etc.

    Returns:
        delta (ndarray): State derivatives (37,).
        yp (ndarray): Process measurements (52,).
        fcc_e_new (ndarray): Updated PID error vector (6,).
    """
    # Unpack disturbances
    Tatm, API, T1, _, _, _, d7, d8, d9 = dist[:9]

    # Unpack manipulated variables and setpoints
    ufcc = params['initial']['ufcc']
    # [F3, F4, V12, V13, V14, V15, Vlift, V5, V16,
    #  SPPH, SPMF, SPRG, SPRGT, SPCI, SPRT]
    *mvs, SPPH, SPMF, SPRG, SPRGT, SPCI, SPRT = ufcc

    Flpg       = params['initial']['Flpg']
    Tcondenser = params['initial']['Tcondenser']
    variance   = params['fccVariance']
    old_errors = params['fcc_e_old']
    dt         = params['dt']

    # Initialize PID controllers with previous errors
    Kc = params['FCC_PID']['Kc']
    Tao = params['FCC_PID']['Tao']
    pc2 = PIDController(Kc[0], Tao[0], dt=dt)
    tc3 = PIDController(Kc[1], Tao[1], dt=dt)
    pc1 = PIDController(Kc[2], Tao[2], dt=dt)
    lc1 = PIDController(Kc[3], Tao[3], dt=dt)
    tc1 = PIDController(Kc[4], Tao[4], dt=dt)
    tc2 = PIDController(Kc[5], Tao[5], dt=dt)
    # Restore previous integral/error
    for pid, err in zip([pc2, tc3, pc1, lc1, tc1, tc2], old_errors):
        pid.prev_error = err

    # Unpack states
    (T3, T2, P7, P5, P3, P6, rho, P2,
     Csc, Crgc, Treg, Wsp, Wreg, Rn, Wr,
     Tr, Fair, Pblp, P1, PreHeatE, MainFracE,
     RegenE, RegenTE, CatIE, ReacTE,
     GFpvgo, GFp1, GFp2, GFp3, GFp4,
     GFc5, GFb, GFp, GFe, GFm,
     Fwg, Fcoke) = xfcc

    # Calculate control moves
    # PC2: fractionator pressure control, SP=SPMF, PV=P5
    V4 = pc2.update(SPMF, P5)
    if d7 != 0: V4 = d7

    # TC3: regenerator temperature, SP=SPRGT, PV=Treg
    p6 = tc3.update(SPRGT, Treg)
    if d8 != 0: p6 = d8

    # PC1: regenerator pressure, SP=SPRG, PV=P6
    V7 = pc1.update(SPRG, P6)
    if d9 != 0: V7 = d9

    # LC1: catalyst inventory level, SP=SPCI, PV=Wr
    V3 = lc1.update(SPCI, Wr)

    # TC1: feed preheater temperature, SP=SPPH, PV=T2
    V1 = tc1.update(SPPH, T2)

    # TC2: reactor temperature, SP=SPRT, PV=Tr
    V2 = tc2.update(SPRT, Tr)

    # Placeholder: secondary flows (FV11, F11, etc.) and PFR
    # e.g., FV11 = ...
    # Placeholder: Kinetic call
    # a0 = np.zeros(11)
    # y_pfr, beta = PFR(a0, F3, F4, (F3+F4)/rho, API, Tr, P5, Crgc, Wr)

    # Compute derivatives (stubbed)
    delta = np.zeros(37)

    # Compute measurements (stubbed)
    yp = np.zeros(52)

    # Collect updated errors
    fcc_e_new = np.array([pc2.prev_error, tc3.prev_error,
                           pc1.prev_error, lc1.prev_error,
                           tc1.prev_error, tc2.prev_error])

    return delta, yp, fcc_e_new
