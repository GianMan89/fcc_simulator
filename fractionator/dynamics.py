"""
fractionator/dynamics.py

Dynamic fractionator (DSAF) ported from Fractionator.m and Fractionatori.m.
Implements stage-by-stage MESH balances with flash solves and four PID controllers.
"""
import numpy as np
from scipy.optimize import fsolve
from thermodynamics.enthalpy import enthalpy, enthalpyB
from utils.controllers import PIDController


def mesh_flash(z, H, V, x_old, y_old, xc_stage, MW, P, T_stage):
    """
    Solve MESH equations for one stage flash:
      M: material balance
      E: energy balance
      S: equilibrium relations
    z: feed composition (vector)
    H, V: liquid and vapor molar rates
    x_old, y_old: initial guesses for liquid & vapor compositions
    xc_stage: liquid composition from internal profile
    MW: molecular weights
    P: stage pressure
    T_stage: stage temperature
    Returns: x, y (compositions), HL, HV
    """
    k = len(z)
    def residuals(vars):
        x = vars[:k]
        y = vars[k:2*k]
        HL, HV = enthalpyB(T_stage, P, x, MW), enthalpy(T_stage, P, x, MW)[3]
        # Material balance: z*F = x*L + y*V
        # Energy balance: z*Hf*F = x*HL*L + y*HV*V + Q
        # Equilibrium: y = K*x
        # Placeholder residuals: implement full MESH per stage
        res = np.zeros(2*k+1)
        res[:k] = z* (H+V) - x*H - y*V
        res[k:2*k] = y - x  # ideal Raoult, placeholder
        res[-1] = np.sum(x) - 1.0
        return res
    guess = np.concatenate([x_old, y_old])
    sol = fsolve(residuals, guess)
    x = sol[:k]
    y = sol[k:2*k]
    HL, _ = enthalpyB(T_stage, P, x, MW)
    _, _, _, _, HV = enthalpy(T_stage, P, x, MW)
    return x, y, HL, HV


def fractionator_dynamics(xfra, ufra, xfil, params):
    """
    Computes derivatives for fractionator states and manipulated vars.

    xfra: state vector [Hold ups (n); EnL (n); EnV (n); Liq flows (n)]
    ufra: manipulated vars [Vap flow, Condenser duty]
    xfil: filtered FCC outputs feeding column
    params: dict with initial MV, SP, products, errord, xc, MW, P_profile, controllers, flashDelta

    Returns: dxfra, dufra
    """
    n = params['Nstages']
    MW = params['Mwin']
    P_profile = params['P_profile']  # array of stage pressures length n
    T_profile = params['T_profile']  # initial stage temps

    # Unpack state
    Hold = xfra[0:n]
    EnL  = xfra[n:2*n]
    EnV  = xfra[2*n:3*n]
    Liq  = xfra[3*n:4*n]

    # Setpoints and product flows
    SP       = params['initial']['SP']
    products = params['initial']['products']

    # Controllers for condenser duty and vap flow
    level_pid = PIDController(*params['L1_PID'], dt=params['flashDelta']*3600,
                               u_min=params['valve_min_pct'], u_max=params['valve_max_pct'])
    temp_pid  = PIDController(*params['TC_PID'], dt=params['flashDelta']*3600,
                               u_min=params['valve_min_pct'], u_max=params['valve_max_pct'])
    hn_pid    = PIDController(*params['HN_PID'], dt=params['flashDelta']*3600,
                               u_min=params['valve_min_pct'], u_max=params['valve_max_pct'])
    lco_pid   = PIDController(*params['LCO_PID'], dt=params['flashDelta']*3600,
                               u_min=params['valve_min_pct'], u_max=params['valve_max_pct'])

    newHold = np.zeros(n)
    newEnL  = np.zeros(n)
    newEnV  = np.zeros(n)
    newLiq  = np.zeros(n)

    # Initial guesses for compositions
    x_old = np.tile(params['initial']['xc'][0,:], (n,1)).T
    y_old = x_old.copy()

    # Loop through stages (could alternate direction)
    for i in range(n):
        # Feed composition: from xfil for stage 1 else from previous stage
        if i==0:
            z = xfil[:len(MW)]
        else:
            z = newx  # from previous stage
        L = Liq[i]
        V = ufra[0]  # placeholder vap flow
        P_stage = P_profile[i]
        T_stage = T_profile[i]

        # Flash solve
        x_sol, y_sol, HL_sol, HV_sol = mesh_flash(z, L, V, x_old[:,i], y_old[:,i],
                                                  params['initial']['xc'][i,:], MW, P_stage, T_stage)
        # Hold-up dynamic
        newHold[i] = Hold[i] + params['flashDelta'] * (z.sum()*L - y_sol.sum()*V)
        newEnL[i]  = EnL[i]  + params['flashDelta'] * (z.dot(HL_sol)*L - y_sol.dot(HV_sol)*V)
        newEnV[i]  = EnV[i]  + params['flashDelta'] * (z.dot(HL_sol)*L - y_sol.dot(HV_sol)*V)
        newLiq[i]  = L        # placeholder, recompute cut flow

        # Update old guesses
        x_old[:,i] = x_sol
        y_old[:,i] = y_sol
        newx       = x_sol

    # Controller update using setpoints and products
    level_set = SP[0]
    T_set     = SP[1]
    val_hn    = hn_pid.update(SP[2], products[1])
    val_lco   = lco_pid.update(SP[3], products[2])
    vap_flow  = level_pid.update(level_set, newHold[0])
    cond_duty = temp_pid.update(T_set, T_profile[-1])
    newUfra   = np.array([vap_flow, cond_duty, val_hn, val_lco])

    # Derivatives
    dt_s = params['flashDelta']*3600.0
    dxfra = np.concatenate([(newHold-Hold)/dt_s,
                             (newEnL-EnL)/dt_s,
                             (newEnV-EnV)/dt_s,
                             (newLiq-Liq)/dt_s])
    dufra = (newUfra - ufra) / dt_s

    return dxfra, dufra
