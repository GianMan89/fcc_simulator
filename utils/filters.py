"""
utils/filters.py

Provides first-order filter dynamics for smoothing signals.
"""
import numpy as np

def filter_derivative(x, u, tau):
    """
    Compute the time derivative for a first-order filter:
        tau * dx/dt + x = u

    Args:
        x (float or ndarray): Current filter state.
        u (float or ndarray): Current input signal.
        tau (float): Time constant (same units as time in simulation).

    Returns:
        dxdt (float or ndarray): Rate of change of the filter state.
    """
    return (u - x) / tau


def apply_filter_discrete(x, u, tau, dt):
    """
    Discrete-time update of a first-order filter using Euler integration:
        x_next = x + dt * (u - x) / tau

    Args:
        x (float or ndarray): Current filter state.
        u (float or ndarray): Current input signal.
        tau (float): Time constant.
        dt (float): Time step.

    Returns:
        x_next (float or ndarray): Updated filter state after dt.
    """
    dxdt = filter_derivative(x, u, tau)
    return x + dxdt * dt
