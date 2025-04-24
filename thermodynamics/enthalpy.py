"""
thermodynamics/enthalpy.py

SRK equation-of-state flash and enthalpy calculations for vapor and liquid,
ported from Enthalpy.m and EnthalpyB.m.
"""
import numpy as np

def enthalpy(T, P, x, Mw):
    """
    Compute equilibrium vapor fraction, fugacity coefficients, and enthalpies.

    Args:
        T (float): Temperature in Kelvin.
        P (float): Pressure in bar.
        x (ndarray): Liquid-phase mole fractions (length k).
        Mw (ndarray): Molecular weights of components (length k).

    Returns:
        y (ndarray): Vapor-phase fractions (length k).
        Zl (float): Liquid compressibility factor.
        K (ndarray): Equilibrium K-factors (length k).
        HL (ndarray): Liquid molar enthalpies (kJ/mol).
        HV (ndarray): Vapor molar enthalpies (kJ/mol).
    """
    R = 8.31445 / 100.0  # bar·L/(K·mol)
    k = len(x)

    # Determine critical properties for pseudo-components
    # First four fixed: methane, ethane, propane, butane
    Tf = np.array([190.6990051, 305.428009, 369.8980103, 425.1990051])
    Pf = np.array([46.40670577, 48.83839996, 42.56651352, 37.96612264])
    wf = np.array([0.01150, 0.09860, 0.15240, 0.20100])

    # Pseudo-components (5...k)
    wpse = np.zeros(k-4)
    Tcpse = np.zeros(k-4)
    Pcpse = np.zeros(k-4)
    for i in range(k-4):
        mw = Mw[i]
        wpse[i] = -5.55305029654881e-07*mw**2 + 0.00236172564924776*mw + 0.0885182765841059
        Tcpse[i] = (1.76502197695905e-11*mw**5 - 4.36256156911342e-08*mw**4
                    + 4.13720500729747e-05*mw**3 - 0.0193203380223777*mw**2
                    + 5.15466244955443*mw + 185.407202849323)
        Pcpse[i] = (-3.94956174670886e-13*mw**5 + 1.18757248039115e-09*mw**4
                    - 1.40250698603397e-06*mw**3 + 0.000827965933144259*mw**2
                    - 0.264543405611834*mw + 48.4998241245831)

    Tc = np.concatenate([Tf, Tcpse])
    Pc = np.concatenate([Pf, Pcpse])
    w_all = np.concatenate([wf, wpse])

    # Calculate ai, bi
    ai = np.zeros(k)
    bi = np.zeros(k)
    for i in range(k):
        Tr_ratio = T / Tc[i]
        m = 0.480 + 1.574*w_all[i] - 0.176*w_all[i]**2
        alpha = (1 + m*(1 - np.sqrt(Tr_ratio)))**2
        ai[i] = 0.42747*(R*Tc[i])**2 * alpha / Pc[i]
        bi[i] = 0.08664*R*Tc[i] / Pc[i]

    # Mixing rules
    b = np.dot(x, bi)
    as_mix = np.dot(x, np.sqrt(ai))
    a_mix = 0.0
    for i in range(k):
        for j in range(k):
            a_mix += x[i]*x[j]*np.sqrt(ai[i]*ai[j])
    A = a_mix*P/(R*T**2)
    B = b*P/(R*T)

    # Cubic EOS solve: Z^3 + c2*Z^2 + c1*Z + c0 = 0
    coeffs = [1, -(1 - B), A - 2*B - 3*B**2, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    Zl = min(real_roots)
    Zv = max(real_roots)

    # Fugacity coefficients
    phi_l = np.exp((bi/b)*(Zl-1) - np.log(Zl-B)
                   - (A/B)*(2*as_mix/a_mix - bi/b)*np.log((Zl+B)/Zl))
    phi_v = np.exp((bi/b)*(Zv-1) - np.log(Zv-B)
                   - (A/B)*(2*as_mix/a_mix - bi/b)*np.log((Zv+B)/Zv))

    # K-factors & vapor composition
    K = phi_l/phi_v
    y = x * K
    y = y / np.sum(y)

    # Heat capacity integration coefficients (Cp = A + B*T + C*T^2)
    # For first 4 components
    Cp_fixed = np.array([[1.702, 9.081e-3, -2.164e-6],
                         [1.131, 19.225e-3, -5.561e-6],
                         [1.213, 28.785e-3, -8.824e-6],
                         [1.935, 36.915e-3, -11.402e-6]])
    Cp_av = Cp_fixed.mean(axis=0)
    Cp_all = np.vstack([Cp_fixed] + [Cp_av]*(k-4))

    # Integrate Cp from 298.15 to T
    def integrate_cp(coeff, T):
        A_cp, B_cp, C_cp = coeff
        return (A_cp*(T-298.15)
                + B_cp*(T**2-298.15**2)/2
                + C_cp*(T**3-298.15**3)/3)

    HL = np.zeros(k)
    HV = np.zeros(k)
    for i in range(k):
        dH = integrate_cp(Cp_all[i], T) * R / 1000  # kJ/mol
        # Standard formation enthalpy
        Hf = np.array([-74.9, -84.738, -103.89, -126.19] + [0]*(k-4))[i]
        H_ideal = Hf + dH
        # Neglect EOS departure here
        HL[i] = H_ideal
        HV[i] = H_ideal

    return y, Zl, K, HL, HV


def enthalpyB(T, P, x, Mw):
    """
    Bottom stage enthalpy calculations (liquid-phase only).
    """
    # Reuse enthalpy to get HL
    _, _, _, HL, _ = enthalpy(T, P, x, Mw)
    return HL
