import numpy as np
import os
import pandas as pd
from oblique_shock_solver import ObliqueShockSolver
from taylor_maccoll_solver import TaylorMaccollSolver

class ConicalFlowAnalyzer:
    def __init__(self, M1, gamma):
        self.M1 = M1
        self.gamma = gamma
        self.os_solver = ObliqueShockSolver(gamma=gamma)
        self.tm_solver = TaylorMaccollSolver(gamma=gamma)

    def solve_taylor_maccoll(self, theta_s_deg):
        """
        Solves for the cone angle and normalized velocity components 
        given a shock angle using the Taylor-Maccoll and Oblique Shock Solvers.

        Parameters:
            theta_s_deg (float): Shock angle in degrees.

        Returns:
            pd.DataFrame: A DataFrame containing theta (cone angle in degrees), V_r, and V_theta for each iteration.
        """
        # Convert shock angle to radians
        theta_s = np.radians(theta_s_deg)

        # Step 1: Use ObliqueShockSolver to calculate post-shock conditions
        oblique_shock_results = self.os_solver.calculate_post_shock_conditions(self.M1, theta_s)
        theta_w = oblique_shock_results["delta"]  # Flow deflection angle
        M2 = oblique_shock_results["M2"]

        # Step 2: Calculate normalized velocity components
        V_prime, V_r, V_theta = self.tm_solver.calculate_velocity_components(M2, theta_s, theta_w)

        # Step 3: Use TaylorMaccollSolver to find the cone angle and iterate
        theta_c, V_r, V_theta = self.tm_solver.solve(theta_w, V_r, V_theta)

        return theta_c, V_r, V_theta
    
    def tabulate_tm_shock_to_cone(self, theta_s_deg):
        """
        Solves from the shock angle to the cone angle using Taylor-Maccoll equations.

        Parameters:
            theta_s_deg (float): Shock angle in degrees.

        Returns:
            pd.DataFrame: A DataFrame containing theta (degrees), V_r, and V_theta.
        """
        theta_s = np.radians(theta_s_deg)
        theta_c_deg, V_r0, V_theta0 = self.solve_taylor_maccoll(theta_s_deg)
        theta_c = np.radians(theta_c_deg)
        df = self.tm_solver.tabulate_from_shock_to_cone(theta_s, theta_c, V_r0, V_theta0)
        print(df)