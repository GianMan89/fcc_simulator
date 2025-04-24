"""
utils/controllers.py

Generic PID controller class for FCC–Fractionator simulation.
"""

import numpy as np

class PIDController:
    """
    Discrete PID controller with P, I, and optional D action plus output limiting.

    Attributes:
        Kp (float): Proportional gain.
        Ti (float): Integral time constant (sec).
        Td (float): Derivative time constant (sec).
        dt (float): Time step (sec).
        u_min (float or None): Minimum output limit.
        u_max (float or None): Maximum output limit.
        integral (float): Accumulated integral term.
        prev_error (float): Previous loop error.
    """
    def __init__(self, Kp, Ti, Td=0.0, dt=1.0, u_min=None, u_max=None):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self, integral=0.0, prev_error=0.0):
        """Reset the controller internal states."""
        self.integral = integral
        self.prev_error = prev_error

    def update(self, setpoint, measurement, feedforward=0.0):
        """
        Compute the PID controller output.

        Args:
            setpoint (float): Desired target value.
            measurement (float): Current measured value.
            feedforward (float): Optional feedforward term.

        Returns:
            u (float): Control output after limiting.
        """
        error = setpoint - measurement
        # Proportional term
        P = self.Kp * error

        # Integral term (trapezoidal approximation)
        if self.Ti != 0:
            self.integral += 0.5 * (error + self.prev_error) * (self.dt / self.Ti)
        I = self.Kp * self.integral

        # Derivative term
        derivative = 0.0
        if self.Td != 0:
            derivative = self.Td * (error - self.prev_error) / self.dt
        D = self.Kp * derivative

        # PID output before limits
        u = feedforward + P + I + D

        # Apply output limits
        if self.u_min is not None or self.u_max is not None:
            u = np.clip(u, self.u_min if self.u_min is not None else u,
                          self.u_max if self.u_max is not None else u)

        # Update state
        self.prev_error = error
        return u
