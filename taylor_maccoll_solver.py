'''
==================================================
File: taylor_maccoll_solver.py
Purpose: Implements the oblique shock solver for compressible flow analysis.

Authors (listed alphabetically)
------------------------------------
Benjamin Lasher
Courtney Paternak
Dominic Perito
Juan P. Roldan

Last Updated: 1/16/2025

Nomenclature:
    gamma   : Specific heat ratio (dimensionless), default is 1.4 for air
    M1      : Freestream Mach number upstream of the shock
    theta_s : Shock wave angle (radians)
    delta   : Flow deflection angle (radians)
    M2      : Downstream Mach number
    Vr      : Radial velocity component (normalized)
    Vtheta  : Normal velocity component (normalized)
==================================================
'''

import numpy as np
from oblique_shock_solver import ObliqueShockSolver

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, Vmax=1.0, step_size=0.01):
        '''
            Initializes the Taylor-Maccoll solver with default parameters.

            Parameters
            ----------
            gamma : float
                Specific heat ratio, default is 1.4 for air.
            Vmax : float
                Maximum velocity (normalized).
            step_size : float
                Integration step size in radians.
        '''
        self.gamma = gamma
        self.Vmax = Vmax
        self.h = step_size

    def taylor_maccoll_system(self, theta, Vr, dVr):
        '''
            Defines the Taylor-Maccoll ODE system.

            Parameters
            ----------
            theta : float
                Angle of the position vector from the cone vertex (radians).
            Vr : float
                Radial component of velocity (normalized).
            dVr : float
                First derivative of Vr with respect to theta.

            Returns
            -------
            np.array
                A 2-element array containing dVr and ddVr.
        '''
        term1 = (self.gamma - 1) / 2 * (self.Vmax**2 - Vr**2 - dVr**2)
        term2 = (2 * Vr + dVr * np.cot(theta))
        numerator = term1 * term2 - dVr * (Vr * dVr + dVr**2)
        denominator = term1 - Vr * dVr
        ddVr = numerator / denominator
        return np.array([dVr, ddVr])

    def rk4_step(self, theta, Vr, dVr):
        '''
            Performs a single RK4 integration step.

            Parameters
            ----------
            theta : float
                Current angle (radians).
            Vr : float
                Current radial velocity.
            dVr : float
                Current derivative of radial velocity.

            Returns
            -------
            tuple
                Updated values of Vr and dVr after one step.
        '''
        k1 = self.h * self.taylor_maccoll_system(theta, Vr, dVr)
        k2 = self.h * self.taylor_maccoll_system(theta + self.h / 2, Vr + k1[0] / 2, dVr + k1[1] / 2)
        k3 = self.h * self.taylor_maccoll_system(theta + self.h / 2, Vr + k2[0] / 2, dVr + k2[1] / 2)
        k4 = self.h * self.taylor_maccoll_system(theta + self.h, Vr + k3[0], dVr + k3[1])
        dVr_next = dVr + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        Vr_next = Vr + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        return Vr_next, dVr_next

    def solve(self, theta0, Vr0, dVr0, theta_end):
        '''
            Solves the Taylor-Maccoll equation over a range of angles.

            Parameters
            ----------
            theta0 : float
                Initial angle (radians).
            Vr0 : float
                Initial radial velocity.
            dVr0 : float
                Initial derivative of Vr.
            theta_end : float
                Final angle (radians).

            Returns
            -------
            tuple
                Arrays of theta, Vr, and dVr.
        '''
        theta_values = [theta0]
        Vr_values = [Vr0]
        dVr_values = [dVr0]

        theta = theta0
        Vr = Vr0
        dVr = dVr0

        while theta < theta_end:
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h
            theta_values.append(theta)
            Vr_values.append(Vr)
            dVr_values.append(dVr)

        return np.array(theta_values), np.array(Vr_values), np.array(dVr_values)