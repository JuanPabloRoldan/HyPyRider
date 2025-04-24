import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from isentropic_relations_solver import IsentropicRelationsSolver

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, step_size=0.00005):
        self.gamma = gamma
        self.h = step_size  # Angular step size (radians)
        self.gas_const = 287  # Specific gas constant (J/kg-K)
        self.temp_static = 293  # Static temperature in Kelvin
        self.theta_s_deg = 35.0  # Shock angle in degrees
        self.M3 = 2.273  # Freestream Mach number

    def compute_post_shock_mach_components(self):
        """Compute normal Mach numbers immediately before and after the shock."""
        beta = np.radians(self.theta_s_deg)
        Mn3 = self.M3 * np.sin(beta)
        num = Mn3**2 + self.gamma / (self.gamma - 1)
        den = Mn3**2 * (2 * self.gamma / (self.gamma - 1)) - 1
        Mn2 = np.sqrt(num / den)
        return Mn2, Mn3

    def compute_deflection_and_postshock_mach(self, Mn2, tol=1e-5, max_iter=100):
        """Compute the shock deflection angle and postshock Mach number."""
        Delta = np.radians(5)
        beta = np.radians(self.theta_s_deg)
        iteration = 0
        while iteration < max_iter:
            Beta = beta + Delta
            M2 = Mn2 / np.sin(Beta)
            num = Mn2**2 - 1
            den = Mn2**2 * (self.gamma + np.cos(2 * Beta) + 2)
            Delta_new = np.arctan(2 * (1 / np.tan(Beta)) * (num / den))
            if abs((Delta_new - Delta) / Delta_new) < tol:
                break
            Delta = Delta_new
            iteration += 1
        return Delta, M2, iteration

    def compute_initial_velocity_components(self, Mn2, M2):
        """Compute and normalize initial radial and tangential velocity components post-shock."""
        T0 = self.temp_static * (1 + (self.gamma - 1)/2 * M2**2)
        T2 = T0 / (1 + (self.gamma - 1)/2 * M2**2)
        a2 = np.sqrt(self.gamma * self.gas_const * T2)
        V_theta = -Mn2 * a2
        V_r = np.sqrt(M2**2 - Mn2**2) * a2
        return V_r / a2, V_theta / a2

    def compute_mach(self, V_r, V_theta):
        """Compute local Mach number from normalized velocity components."""
        return np.sqrt(V_r**2 + V_theta**2)

    def taylor_maccoll_rhs(self, theta, Vr, dVr):
        B = (self.gamma - 1) / 2 * (1 - Vr**2 - dVr**2)
        C = (2 * Vr + dVr / np.tan(theta))
        ddVr = (dVr**2 - B * C) / (B - dVr**2)
        return np.array([dVr, ddVr])

    def rk4_step(self, theta, Vr, dVr):
        """Perform a single Runge-Kutta 4th order integration step."""
        K1, M1 = self.taylor_maccoll_rhs(theta, Vr, dVr)
        K1 *= self.h
        M1 *= self.h

        K2 = self.h * (dVr + 0.5 * M1)
        M2 = self.h * self.taylor_maccoll_rhs(theta + 0.5 * self.h, Vr + 0.5 * K1, dVr + 0.5 * M1)[1]

        K3 = self.h * (dVr + 0.5 * M2)
        M3 = self.h * self.taylor_maccoll_rhs(theta + 0.5 * self.h, Vr + 0.5 * K2, dVr + 0.5 * M2)[1]

        K4 = self.h * (dVr + M3)
        M4 = self.h * self.taylor_maccoll_rhs(theta + self.h, Vr + K3, dVr + M3)[1]

        Vr_next = Vr + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        dVr_next = dVr + (1 / 6) * (M1 + 2 * M2 + 2 * M3 + M4)
        return Vr_next, dVr_next

    def solve_flow_field(self, theta_s, theta_c, Vr0, M_init, dVr0):
        """Integrate the Taylor-Maccoll equations from shock to cone surface."""
        isentropic_solver = IsentropicRelationsSolver(self.gamma)
        theta = theta_c
        Vr = Vr0
        dVr = dVr0
        results = []

        while abs(theta - theta_s) > 1e-3:
            try:
                M = self.compute_mach(Vr, dVr)
                props = isentropic_solver.isentropic_relations(M)
                if isinstance(props, str):
                    break
                p_ratio = props["Static Pressure Ratio (p/p0)"]
                t_ratio = props["Static Temperature Ratio (T/T0)"]
                rho_ratio = props["Static Density Ratio (rho/rho0)"]
            except Exception:
                break

            results.append([theta, M, Vr, dVr, p_ratio, t_ratio, rho_ratio])
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            theta += self.h

        return pd.DataFrame(results, columns=[
            "Theta (radians)", "Mach", "V_r", "V_theta", "P/P0", "T/T0", "rho/rho0"
        ])

if __name__ == "__main__":
    solver = TaylorMaccollSolver()
    theta_s = np.radians(solver.theta_s_deg)
    theta_c = theta_s - np.radians(10)

    Mn2, Mn3 = solver.compute_post_shock_mach_components()
    Delta, M2, _ = solver.compute_deflection_and_postshock_mach(Mn2)
    V_r, V_theta = solver.compute_initial_velocity_components(Mn2, solver.M3)
    df = solver.solve_flow_field(theta_s, theta_c, V_r, M2, dVr0=1)

    print(df.head())

    # Integrate inlet shape using dr/dθ = V_θ / V_r
    theta_vals = df["Theta (radians)"].to_numpy()
    V_r_vals = df["V_r"].to_numpy()
    V_theta_vals = df["V_theta"].to_numpy()

    r_vals = [1.0]
    for i in range(1, len(theta_vals)):
        dtheta = theta_vals[i] - theta_vals[i - 1]
        slope = V_theta_vals[i - 1] / V_r_vals[i - 1]
        r_vals.append(r_vals[-1] + slope * dtheta)

    x = [r * np.sin(theta) for r, theta in zip(r_vals, theta_vals)]
    y = [r * np.cos(theta) for r, theta in zip(r_vals, theta_vals)]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Inlet Wall Contour")
    plt.xlabel("x")
    plt.ylabel("r")
    plt.title("Busemann Inlet Shape (from Taylor-Maccoll)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(theta_vals), df["Mach"].to_numpy(), label="Mach Number")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Mach Number")
    plt.title("Taylor-Maccoll Supersonic Flow Field")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()