'''
==================================================
Embry-Riddle Aeronautical University
AE595CC Hypersonic Vehicle Design

Project 01: Conical Flow Analyzer

Authors (listed alphabetically)
------------------------------------
Benjamin Lasher
Courtney Paternak
Dominic Perito
Juan P. Roldan

Last Updated: 1/23/2025

Description:
    This project models the conical flow field using the Taylor-Maccoll equation,
    solving for post-shock conditions using the oblique shock relations.

Nomenclature:
    gamma   : Specific heat ratio (dimensionless), default is 1.4 for air
    M1      : Freestream Mach number upstream of the shock
    theta_s : Shock wave angle (radians)
    delta   : Flow deflection angle (radians)
    M2      : Downstream Mach number
    Vr      : Radial velocity component (normalized)
    Vtheta  : Normal velocity component (normalized)

Usage:
    Run this script to solve for conical flow parameters given upstream conditions.
==================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from HyPyRider.ConicalFlowAnalyzer.oblique_shock_solver import ObliqueShockSolver
from HyPyRider.ConicalFlowAnalyzer.taylor_maccoll_solver import TaylorMaccollSolver


class ConicalFlowAnalyzer:
    def __init__(self, gamma=1.4, M1=10.0):
        """
        Initializes the Conical Flow Analyzer.

        Parameters:
            gamma (float): Specific heat ratio for the fluid.
            M1 (float): Freestream Mach number upstream of the shock.
        """
        self.gamma = gamma
        self.M1 = M1
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
        theta_values, V_r, V_theta = self.tm_solver.solve(delta, V_r, V_theta)
        cone_angle = np.degrees(theta_values[-1])  # Final theta is the cone angle

        return cone_angle, V_r, V_theta

    def solve__post_shock_mach(self, V_r, V_theta):
        """
        Solves for the Mach of the flow from the normalized velocity components.

        Parameters:
            V_r (float): Normalized radial velocity.
            V_theta (float): Normalized tangential velocity.

        Returns:
            float: Mach number.
        """

        V = np.sqrt(V_r ** 2 + V_theta ** 2)
        M = np.sqrt((2 / (self.gamma - 1)) * (V ** 2 / (1 - V ** 2)))

        return M

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

    def plot_shock_vs_cone(self, data):
        """
        Plots the shock angle vs. cone angle using a given DataFrame.

        Parameters:
            data (pd.DataFrame): DataFrame containing "Shock Angle (degrees)" and "Cone Angle (degrees)" columns.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data["Cone Angle (degrees)"], data["Shock Angle (degrees)"], label=f"Mach = {self.M1}")
        plt.xlabel("Cone Angle (degrees)")
        plt.ylabel("Shock Angle (degrees)")
        plt.title(f"Shock Angle vs Cone Angle at γ = {self.gamma}")
        plt.grid()
        plt.legend()
        plt.show()