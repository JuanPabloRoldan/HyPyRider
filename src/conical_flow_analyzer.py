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

    def solve_taylor_maccoll_full(self, theta_s_deg):
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
        delta = oblique_shock_results["delta"]  # Flow deflection angle
        M2 = oblique_shock_results["M2"]

        # Step 2: Calculate normalized velocity components
        V_prime, V_r, V_theta = self.tm_solver.calculate_velocity_components(M2, theta_s, delta)

        # Step 3: Use TaylorMaccollSolver to find the cone angle and iterate
        results_df = self.tm_solver.solve(delta, V_r, V_theta)

        # Prepare data for output
        results_df = pd.DataFrame(results_df, columns=["Theta (degrees)", "V_r", "V_theta"])

        return results_df

    def solve_taylor_maccoll(self, theta_s_deg):
        """
        Solves for the cone angle and normalized velocity components 
        and returns only the final values (cone angle, V_r, V_theta).

        Parameters:
            theta_s_deg (float): Shock angle in degrees.

        Returns:
            tuple: Final cone angle (degrees), V_r, and V_theta as floats.
        """
        # Solve for the full table
        results_df = self.solve_taylor_maccoll_full(theta_s_deg)

        # Get the last row
        last_row = results_df.iloc[-1]

        # Extract the last three values and return as a tuple
        return last_row["Theta (degrees)"], last_row["V_r"], last_row["V_theta"]