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

Last Updated: 1/26/2025

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

    def calculate_flow_deflection_angle(self, M1, theta_s):
        '''
            Calculates the flow deflection angle (delta).

            Parameters
            ----------
            M1 : float
                Freestream Mach number upstream of the shock.
            theta_s : float
                Shock wave angle in radians.

            Returns
            -------
            float
                Flow deflection angle (delta) in radians.
        '''
        cot_delta = np.tan(theta_s) * (((self.gamma + 1) * M1 ** 2) / (2 * (M1 ** 2 * (np.sin(theta_s) ** 2) - 1)) - 1)
        delta = np.arctan(1 / cot_delta)
        return delta

    def calculate_post_shock_conditions(self, M1, theta_s):
        """
        Calculates the post-shock conditions, including the downstream Mach number (M2), 
        flow deflection angle (delta), and thermodynamic property ratios.

        Parameters
        ----------
        M1 : float
            Freestream Mach number upstream of the shock.
        theta_s : float
            Shock wave angle in radians.

        Returns
        -------
        dict
            {
                "delta": float
                    Flow deflection angle in radians.
                "M2": float
                    Downstream Mach number after the shock.
                "P2_P1": float
                    Pressure ratio across the shock (P2/P1).
                "rho2_rho1": float
                    Density ratio across the shock (rho2/rho1).
                "T2_T1": float
                    Temperature ratio across the shock (T2/T1).
            }
        """
        # Compute the flow deflection angle (delta)
        delta = self.calculate_flow_deflection_angle(M1, theta_s)

        # Compute the normal component of the upstream Mach number
        M1_normal = M1 * np.sin(theta_s)

        # Compute the normal component of the downstream Mach number (M2_normal)
        M2_normal_squared = (1 + ((self.gamma - 1) / 2) * M1_normal**2) / \
                            (self.gamma * M1_normal**2 - 0.5 * (self.gamma - 1))
        M2_normal = np.sqrt(M2_normal_squared)

        # Compute the full downstream Mach number
        M2 = M2_normal / np.sin(theta_s - delta)

        # Compute the pressure ratio (P2/P1)
        P2_P1 = 1 + (2 * self.gamma / (self.gamma + 1)) * (M1_normal**2 - 1)

        # Compute the density ratio (rho2/rho1)
        rho2_rho1 = (self.gamma + 1) * M1_normal**2 / ((self.gamma - 1) * M1_normal**2 + 2)

        # Compute the temperature ratio (T2/T1)
        T2_T1 = P2_P1 / rho2_rho1

        return {
            "delta": delta,
            "M2": M2,
            "P2_P1": P2_P1,
            "rho2_rho1": rho2_rho1,
            "T2_T1": T2_T1
        }