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
    This project (*SO FAR*) models the conical flow field using the Taylor-Maccoll equation,
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

from oblique_shock_solver import ObliqueShockSolver
from taylor_maccoll_solver import TaylorMaccollSolver
import numpy as np
import matplotlib.pyplot as plt

# Define input parameters
gamma = 1.4  # Specific heat ratio for air
M1 = 3.0  # Freestream Mach number
theta_s_deg = 30  # Shock wave angle in degrees
theta_s = np.radians(theta_s_deg)  # Convert to radians
theta_end_deg = 90  # End angle in degrees
theta_end = np.radians(theta_end_deg)  # Convert to radians

# Step 1: Initialize the Oblique Shock Solver
print("Initializing oblique shock solver...")
os_solver = ObliqueShockSolver(gamma=gamma)
oblique_shock_results = os_solver.calculate_post_shock_conditions(M1, theta_s)

# Extract results from the oblique shock solver
M2 = oblique_shock_results["M2"]
delta = oblique_shock_results["delta"]
Vr0 = oblique_shock_results["Vr"]
dVr0 = oblique_shock_results["Vtheta"]

# Print post-shock conditions
print(f"Post-shock Mach number (M2): {M2:.4f}")
print(f"Flow deflection angle (delta): {np.degrees(delta):.4f} degrees")
print(f"Radial velocity (Vr): {Vr0:.4f}")
print(f"Normal velocity (dVr): {dVr0:.4f}")

# Step 2: Solve the Taylor-Maccoll equation
print("Solving the Taylor-Maccoll equation...")
tm_solver = TaylorMaccollSolver(gamma=gamma)
theta_values, Vr_values, dVr_values = tm_solver.solve(
    theta0=theta_s, Vr0=Vr0, dVr0=dVr0, theta_end=theta_end
)

# Step 3: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta_values), Vr_values, label="Radial Velocity (Vr)")
plt.plot(np.degrees(theta_values), dVr_values, label="Normal Velocity Derivative (dVr)")
plt.xlabel("Theta (degrees)")
plt.ylabel("Velocity (normalized)")
plt.title("Taylor-Maccoll Solution: Velocity Profiles")
plt.legend()
plt.grid()
plt.show()

# Step 4: Save results to a CSV file
output_file = "conical_flow_results.csv"
np.savetxt(output_file, np.column_stack((np.degrees(theta_values), Vr_values, dVr_values)),
           delimiter=",", header="Theta (degrees),Vr (radial),dVr (normal)", comments="")
print(f"Results saved to {output_file}.")