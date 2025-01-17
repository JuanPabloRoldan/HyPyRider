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

Last Updated: 1/16/2025

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
from oblique_shock_solver import ObliqueShockSolver
from taylor_maccoll_solver import TaylorMaccollSolver

def solve_cone_angle(M1, gamma, theta_s_deg):
    """
    Solves for the cone angle given a shock angle using the Taylor-Maccoll and Oblique Shock Solvers.

    Parameters:
        M1 (float): Freestream Mach number upstream of the shock.
        gamma (float): Specific heat ratio for the fluid.
        theta_s_deg (float): Shock angle in degrees.

    Returns:
        float: Cone angle in degrees.
    """
    # Convert shock angle to radians
    theta_s = np.radians(theta_s_deg)

    # Initialize solvers
    os_solver = ObliqueShockSolver(gamma=gamma)
    tm_solver = TaylorMaccollSolver(gamma=gamma)

    # Step 1: Use ObliqueShockSolver to calculate post-shock conditions
    oblique_shock_results = os_solver.calculate_post_shock_conditions(M1, theta_s)
    delta = oblique_shock_results["delta"]  # Flow deflection angle
    V_r = oblique_shock_results["V_r"]
    V_theta = oblique_shock_results["V_theta"]

    # Step 2: Use TaylorMaccollSolver to find the cone angle
    theta_values, _, _ = tm_solver.solve(delta, V_r, V_theta)
    cone_angle = np.degrees(theta_values[-1])  # Final theta is the cone angle

    return cone_angle

def plot_shock_vs_cone(M1, gamma, theta_s_range):
    """
    Plots the shock angle vs. cone angle for a range of shock angles.

    Parameters:
        M1 (float): Freestream Mach number upstream of the shock.
        gamma (float): Specific heat ratio for the fluid.
        theta_s_range (array-like): Array of shock angles in degrees.
    """
    cone_angles = []

    for theta_s_deg in theta_s_range:
        try:
            cone_angle = solve_cone_angle(M1, gamma, theta_s_deg)
            cone_angles.append(cone_angle)
        except ValueError:
            cone_angles.append(None)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(cone_angles, theta_s_range, label=f"Mach = {M1}")
    plt.xlabel("Cone Angle (degrees)")
    plt.ylabel("Shock Angle (degrees)")
    plt.title("Shock Angle vs Cone Angle")
    plt.grid()
    plt.legend()
    plt.show()

# Example Usage
if __name__ == "__main__":
    gamma = 1.2  # Specific heat ratio for air
    M1 = 10.0  # Freestream Mach number
    theta_s_deg = 30  # Shock wave angle in degrees

    # Solve for a single cone angle
    cone_angle = solve_cone_angle(M1, gamma, theta_s_deg)
    print(f"Given Shock Angle: {theta_s_deg}°")
    print(f"Calculated Cone Angle: {cone_angle:.4f}°")

    # Generate plot for a range of shock angles
    theta_s_range = np.linspace(6, 73, 200)  # Exclude 0 and 90 degrees
    plot_shock_vs_cone(M1, gamma, theta_s_range)
