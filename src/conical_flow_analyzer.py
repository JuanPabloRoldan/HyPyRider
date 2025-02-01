import numpy as np
import os
import pandas as pd
from oblique_shock_solver import ObliqueShockSolver
from taylor_maccoll_solver import TaylorMaccollSolver

class ConicalFlowAnalyzer:
    def __init__(self, M1, gamma):
        """
        Initializes the ConicalFlowAnalyzer with Mach number and specific heat ratio.

        Parameters:
            M1 (float): Freestream Mach number upstream of the shock.
            gamma (float): Specific heat ratio for the fluid.
        """
        self.M1 = M1
        self.gamma = gamma
        self.os_solver = ObliqueShockSolver(gamma=gamma)
        self.tm_solver = TaylorMaccollSolver(gamma=gamma)

    def solve_taylor_maccoll(self, theta_s):
        """
        Solves for the cone angle and normalized velocity components 
        given a shock angle using the Taylor-Maccoll and Oblique Shock Solvers.

        Parameters:
            theta_s (float): Shock angle in radians.

        Returns:
            tuple: A tuple containing:
                - theta_c (float): Cone angle in radians.
                - V_r (float): Radial velocity component (normalized).
                - V_theta (float): Tangential velocity component (normalized).
        """

        # Step 1: Use ObliqueShockSolver to calculate post-shock conditions
        oblique_shock_results = self.os_solver.calculate_post_shock_conditions(self.M1, theta_s)
        theta_w = oblique_shock_results["delta"]  # Flow deflection angle
        M2 = oblique_shock_results["M2"]

        # Step 2: Calculate normalized velocity components
        V_prime, V_r, V_theta = self.tm_solver.calculate_velocity_components(M2, theta_s, theta_w)

        # Step 3: Use TaylorMaccollSolver to find the cone angle and iterate
        theta_c, V_r, V_theta = self.tm_solver.solve(theta_w, V_r, V_theta)

        return theta_c, V_r, V_theta
    
    def tabulate_tm_shock_to_cone(self, theta_s):
        """
        Solves from the shock angle to the cone angle using Taylor-Maccoll equations
        and generates a table of results.

        Parameters:
            theta_s (float): Shock angle in radians.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - Theta (radians): Cone angle in radians.
                - V_r: Radial velocity component (normalized).
                - V_theta: Tangential velocity component (normalized).
        """
        # Solve for cone angle and initial velocity components
        theta_c, V_r0, V_theta0 = self.solve_taylor_maccoll(theta_s)

        # Tabulate results from shock to cone
        df = self.tm_solver.tabulate_from_shock_to_cone(theta_s, theta_c, V_r0, V_theta0)
        return df