'''
==================================================
File: oblique_shock_solver.py
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

class ObliqueShockSolver:
    def __init__(self, gamma=1.4):
        '''
            Initializes the oblique shock solver with a specific heat ratio.

            Parameters
            ----------
            gamma : float
                Specific heat ratio, default is 1.4 for air.
        '''
        self.gamma = gamma

    def calculate_post_shock_conditions(self, M1, theta_s):
        '''
            Calculates the post-shock velocity magnitude (V') and its components (V'_r and V'_theta).

            Parameters
            ----------
            M1 : float
                Freestream Mach number upstream of the shock.
            theta_s : float
                Shock wave angle in radians.

            Returns
            -------
            dict
                A dictionary with the following keys:
                - M2: Downstream Mach number.
                - V_prime: Normalized velocity magnitude (V').
                - delta: Flow deflection angle (radians).
                - V_r: Radial component of velocity (V'_r, normalized).
                - V_theta: Normal component of velocity (V'_theta, normalized).
        '''
        if theta_s <= 0 or theta_s >= np.pi / 2:
            raise ValueError("Shock angle must be between 0 and 90 degrees (exclusive).")

        # Calculate the flow deflection angle (delta)
        tan_delta = (2 * np.tan(theta_s) * (M1**2 * np.sin(theta_s)**2 - 1)) / \
                    (M1**2 * (self.gamma + np.cos(2 * theta_s)) + 2)
        delta = np.arctan(tan_delta)

        # Calculate post-shock Mach number (M2)
        M2_normal_squared = 1 / ((self.gamma - 1) / 2 + 1 / (M1**2 * np.sin(theta_s)**2))
        M2 = np.sqrt(M2_normal_squared) / np.sin(theta_s - delta)

        # Calculate V' (normalized velocity magnitude)
        V_prime = (2 / ((self.gamma - 1) * M2**2) + 1) ** -0.5

        # Decompose V' into radial and normal components
        V_r = V_prime * np.cos(delta)
        V_theta = V_prime * np.sin(delta)

        return {
            "M2": M2,
            "V_prime": V_prime,
            "delta": delta,
            "V_r": V_r,
            "V_theta": V_theta
        }