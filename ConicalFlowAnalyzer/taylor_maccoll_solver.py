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

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, step_size=0.0001):
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
        B = (self.gamma - 1) / 2 * (1 - Vr**2 - dVr**2)
        C = (2 * Vr + dVr / np.tan(theta))
        numerator = dVr **2 - (B * C)
        denominator = B - dVr ** 2
        ddVr = numerator / denominator
        return np.array([dVr, ddVr])

    def rk4_step(self, theta, Vr, dVr):
        '''
            Performs a single RK4 integration step for Taylor-Maccoll equations.

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
        # K1 and M1
        K1, M1 = self.taylor_maccoll_system(theta, Vr, dVr)

        # K2 and M2
        K2, M2 = self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * self.h * K1, 
            dVr + 0.5 * self.h * M1
        )

        # K3 and M3
        K3, M3 = self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * self.h * K2, 
            dVr + 0.5 * self.h * M2
        )

        # K4 and M4
        K4, M4 = self.taylor_maccoll_system(
            theta + self.h, 
            Vr + self.h * K3, 
            dVr + self.h * M3
        )

        # Update Vr and dVr
        Vr_next = Vr + (self.h / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        dVr_next = dVr + (self.h / 6) * (M1 + 2 * M2 + 2 * M3 + M4)

        return Vr_next, dVr_next

    def solve(self, theta0, Vr0, dVr0):
        '''
            Solves the Taylor-Maccoll equation until the condition dVr/dtheta >= 0.

            Parameters
            ----------
            theta0 : float
                Initial angle (radians).
            Vr0 : float
                Initial radial velocity.
            dVr0 : float
                Initial derivative of Vr.

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

        while abs(dVr) > 1e-4:  # Continue until dVr/dtheta >= 0
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h

            theta_values.append(theta)
            Vr_values.append(Vr)
            dVr_values.append(dVr)

        return np.array(theta_values), np.array(Vr_values), np.array(dVr_values)