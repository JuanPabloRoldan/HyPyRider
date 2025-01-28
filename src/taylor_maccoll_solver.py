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
import pandas as pd

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, step_size=0.0001):
        '''
        Initializes the Taylor-Maccoll solver with default parameters.

        Parameters
        ----------
        gamma : float
            Specific heat ratio, default is 1.4 for air.
        step_size : float
            Integration step size in radians.
        '''
        self.gamma = gamma
        self.h = step_size

    def calculate_velocity_components(self, M2, theta_s, delta):
        '''
        Calculates and decomposes the normalized velocity magnitude (V') into its components.

        Parameters
        ----------
        M2 : float
            Downstream Mach number (after the shock).
        theta_s : float
            Shock wave angle in radians.
        delta : float
            Flow deflection angle in radians.

        Returns
        -------
        tuple
            V_prime : float
                Normalized velocity magnitude (V').
            V_r : float
                Radial component of velocity (V'_r, normalized).
            V_theta : float
                Tangential component of velocity (V'_theta, normalized).
        '''
        # Compute the normalized velocity magnitude
        V_prime = (2 / ((self.gamma - 1) * M2**2) + 1) ** -0.5

        # Compute the radial velocity component (V_r)
        V_r = V_prime * np.cos(theta_s - delta)

        # Compute the tangential velocity component (V_theta)
        V_theta = V_prime * np.sin(theta_s - delta)

        # Return the computed values
        return V_prime, V_r, V_theta

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
        numerator = dVr**2 - (B * C)
        denominator = B - dVr**2
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
        K1 = self.h * K1
        M1 = self.h * M1

        # K2 and M2
        K2 = self.h * (dVr + 0.5 * M1)
        M2 = self.h * self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * K1, 
            dVr + 0.5 * M1
        )[1]

        # K3 and M3
        K3 = self.h * (dVr + 0.5 * M2)
        M3 = self.h * self.taylor_maccoll_system(
            theta + 0.5 * self.h, 
            Vr + 0.5 * K2, 
            dVr + 0.5 * M2
        )[1]

        # K4 and M4
        K4 = self.h * (dVr + M3)
        M4 = self.h * self.taylor_maccoll_system(
            theta + self.h, 
            Vr + K3, 
            dVr + M3
        )[1]

        # Update Vr and dVr
        Vr_next = Vr + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        dVr_next = dVr + (1 / 6) * (M1 + 2 * M2 + 2 * M3 + M4)

        return Vr_next, dVr_next

    def solve(self, theta0, Vr0, dVr0):
        '''
        Solves the Taylor-Maccoll equation and returns the final values.

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
            Final values of Theta (degrees), V_r, and V_theta.
        '''
        theta = theta0
        Vr = Vr0
        dVr = dVr0

        while abs(dVr) > 1e-3:  # Continue until abs(dVr/dtheta) < 1e-3
            # Perform RK4 step
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h

        # Return final values
        return np.degrees(theta), Vr, dVr
    
    def tabulate_from_shock_to_cone(self, theta_s, theta_c, Vr0, dVr0):
        '''
        Solves the Taylor-Maccoll equation and returns a DataFrame with results.

        Parameters
        ----------
        theta_s : float
            Shock angle (radians).
        theta_c : float
            Cone angle (radians).
        Vr0 : float
            Initial radial velocity.
        dVr0 : float
            Initial derivative of Vr.

        Returns
        -------
        pd.DataFrame
            DataFrame containing Theta (degrees), V_r, and V_theta.
        '''
        theta = theta_c
        Vr = Vr0
        dVr = dVr0

        # Lists to store results
        results = [[np.degrees(theta), Vr, dVr]]  # Log initial conditions

        while not np.isclose(theta, theta_s):  # Continue until reach shock angle
            # Perform RK4 step
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h

            # Save current results (convert theta to degrees)
            results.append([np.degrees(theta), Vr, dVr])

        # Create and return DataFrame
        results_df = pd.DataFrame(results, columns=["Theta (degrees)", "V_r", "V_theta"])
        return results_df