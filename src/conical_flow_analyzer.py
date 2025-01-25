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
            float: Cone angle in degrees.
            float: Normalized radial Velocity
            float: Normalized tangential velocity
        """
        # Convert shock angle to radians
        theta_s = np.radians(theta_s_deg)

        # Step 1: Use ObliqueShockSolver to calculate post-shock conditions
        oblique_shock_results = self.os_solver.calculate_post_shock_conditions(self.M1, theta_s)
        delta = oblique_shock_results["delta"]  # Flow deflection angle
        V_r = oblique_shock_results["V_r"]
        V_theta = oblique_shock_results["V_theta"]

        # Step 2: Use TaylorMaccollSolver to find the cone angle
        theta, V_r, V_theta = self.tm_solver.solve(delta, V_r, V_theta)
        cone_angle = np.degrees(theta)  # Final theta is the cone angle

        return cone_angle, V_r, V_theta

    def solve_taylor_maccoll_range(self, theta_s_range):
        """
        Calculates the shock angle vs. cone angle for a range of shock angles 
        and saves the data to a CSV file.

        Parameters:
            theta_s_range (array-like): Array of shock angles in degrees.

        Returns:
            pd.DataFrame: A DataFrame containing shock angles, cone angles, V_r, and V_theta.
        """
        cone_angles = []
        V_rs = []
        V_thetas = []

        for theta_s_deg in theta_s_range:
            try:
                cone_angle, V_r, V_theta = self.solve_taylor_maccoll(theta_s_deg)
                cone_angles.append(cone_angle)
                V_rs.append(V_r)
                V_thetas.append(V_theta)
            except ValueError:
                cone_angles.append(None)
                V_rs.append(None)
                V_thetas.append(None)

        # Save data to CSV
        data_folder = "data"
        os.makedirs(data_folder, exist_ok=True)
        csv_filename = os.path.join(data_folder, f"ConicalShocks_M{self.M1}_gamma{self.gamma}.csv")
        data = pd.DataFrame({
            "Shock Angle (degrees)": theta_s_range,
            "Cone Angle (degrees)": cone_angles,
            "V_r": V_rs,
            "V_theta": V_thetas
        })
        data.to_csv(csv_filename, index=False)

        print(f"Data saved to {csv_filename}")

        return data