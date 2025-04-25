import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from isentropic_relations_solver import IsentropicRelationsSolver
from scipy.optimize import fsolve

class TaylorMaccollSolver:
    def __init__(self, gamma=1.4, step_size=0.00005):
        '''
        Initializes the Taylor Maccoll Solver with initial parameters

        Parameters
        ----------
        gamma : float
            Specific heat ratio, default is 1.4 for air.
        '''
        self.gamma = gamma
        self.h = step_size  # Angular step size (radians)
        self.gas_const = 287  # Specific gas constant (J/kg-K)
        self.temp_static = 293  # Static temperature in Kelvin
        self.theta_s_deg = 17.2  # Shock angle in degrees og 35
        self.M3 = 2.27  # Freestream Mach number


    def compute_post_shock_mach_components(self):
        """
        Compute normal Mach numbers immediately before and after the shock.

        Parameters
        ----------
            None

        Returns
        ----------
            Mn2 (float): Normal component of M2
            Mn3 (float): Nomral component of M3
        """

        theta = np.radians(self.theta_s_deg)
        Mn3 = self.M3 * np.sin(theta)
        num = Mn3**2 + 2 / (self.gamma + 1)
        den = Mn3**2 * (2 * self.gamma / (self.gamma - 1)) - 1
        Mn2 = np.sqrt(num / den)
        return Mn2, Mn3
    
    def compute_deflection_and_postshock_mach(self, Mn2, tol=1e-5, max_iter=10000):
        """
        Compute the shock deflection angle and postshock Mach number.

        Parameters
        ----------
            Mn2 (float): Normal component of M2

        Returns
        ----------
            Delta (float): Total deflection angle
            M2 (float): Mach number upstream of the shock
            Iteration (int) : iteration value when iteration complete
        """

        deltas = []

        Delta = np.radians(5) #inital geuss
        theta = np.radians(self.theta_s_deg)
        iteration = 0
        while iteration < max_iter:
            Beta = theta + Delta #Breifly set this to radians and it made mach -37.772
            M2 = Mn2 / np.sin(Beta)
            num = Mn2**2 - 1
            den = (Mn2**2 * self.gamma + np.cos(2 * Beta)) * 2 #moved the 2 from + to * 
            Delta_new = np.arctan(2 * (1 / np.tan(Beta)) * (num / den))
            print(Delta_new)
            deltas.append([iteration, Delta_new - Delta])
            if abs((Delta_new - Delta) / Delta_new) < tol:
                break
            Delta = Delta_new
            iteration += 1
        print(iteration)
        print(M2)

        plt.figure(figsize=(8, 6))
        plt.plot([row[0] for row in deltas], [row[1] for row in deltas], label="Delta Convergence")

        M2 = 3 #SETS M2 UNTIL THIS IS FIXED
        print(M2)
        return Delta, M2, iteration

    """
    def compute_deflection_and_postshock_mach(self, Mn2, tol=1e-5, max_iter=1000):

        def theta_beta_mach_eq(Beta, M3, theta_deg, gamma):
            theta = np.radians(theta_deg)
            lhs = np.tan(theta)
            rhs = 2 * (1 / np.tan(Beta)) * ((M3**2 * np.sin(Beta)**2 - 1) / 
                (M3**2 * (gamma + np.cos(2 * Beta)) + 2))
            return lhs - rhs
        
        theta = np.radians(self.theta_s_deg)
        # Initial guess for Beta
        Beta_guess = theta + np.radians(5)

        Beta_solution = fsolve(theta_beta_mach_eq, Beta_guess, args=(self.M3, self.theta_s_deg, self.gamma))[0]
        #M2 = Mn2 / np.sin(Beta_solution)
        Delta = Beta_solution - theta  # Deflection angle

        #ADDED THE FOLLOWING 2 Lines
        Mn2_calculated = self.M3 * np.sin(Beta_solution)
        M2 = Mn2_calculated / np.sin(Beta_solution)
        print(Beta_solution)

        print(M2)
        return Delta, M2, 1
    """

    def compute_initial_velocity_components(self, Mn2, M2):
        """
        Compute and normalize initial radial and tangential velocity components post-shock.

        Parameters
        ----------
            Mn2 (float): Normal component of M2
            M2 (float): Mach number upstream of the shock

        Returns
        ----------
            V_r/a2 (float): Normalized radial component of velocity 
            V_theta/a2(float): Normalized angular component of velocity
        """

        T0 = self.temp_static * (1 + (self.gamma - 1)/2 * M2**2)
        T2 = T0 / (1 + (self.gamma - 1)/2 * M2**2)
        a2 = np.sqrt(self.gamma * self.gas_const * T2)
        V_theta = -Mn2 * a2
        V_r = np.sqrt(M2**2 - Mn2**2) * a2
        return V_r / a2, V_theta / a2

    def compute_mach(self, V_r, V_theta):
        """
        Compute local Mach number from normalized velocity components.

        Parameters
        ----------
            V_r (float): Radial component of velocity
            V_theta (float): Angular component of velocity

        Returns
        ----------
            M (float): Local mach number from vector components.
        """

        M = np.sqrt(V_r**2 + V_theta**2)
        return M

    def taylor_maccoll_rhs(self, theta, Vr, dVr):
        """
        Compute local Mach number from normalized velocity components.

        Parameters
        ----------
            theta (float): cone angle
            Vr (float): Radial component of velocity
            dVr (float): dertivative of Vr with respect to theta
            
        Returns
        ----------
            numpy.ndarray
                A 1D array containing the first and second derivatives of `Vr`:
                [dVr, ddVr], where:
                    - dVr is the input derivative,
                    - ddVr is the second derivative of `Vr` with respect to `theta`.    
        """

        B = (self.gamma - 1) / 2 * (1 - Vr**2 - dVr**2)
        C = (2 * Vr + dVr / np.tan(theta))
        ddVr = (dVr**2 - B * C) / (B - dVr**2)
        return np.array([dVr, ddVr])

    def rk4_step(self, theta, Vr, dVr):
        """
        Perform a single Runge-Kutta 4th order integration step.
        
        Parameters
        ----------
            theta (float): cone angle
            V_r (float): Radial component of velocity
            dVr (float): derivative of V_r

        Returns
        ----------
            Vr_next (): next step of Vr
            dVr_next (): associated derivative of stepped Vr
        
        """

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

    """
    def solve_flow_field(self, theta_s, theta_c, Vr0, M2, dVr0): #replace M_init with M2 still neither are being used.
        ""
        Integrate the Taylor-Maccoll equations from shock to cone surface.

        Parameters
        ----------
            theta_s (float): shock angle
            theta_c (float): cone angle
            Vr0 (float): Vr0
            dVr0 (float): derivative of Vr0

        Returns
        ----------
            results (pd Dataframe): A pd Dataframe of theta, mach, Vr, Vtheta, P/P0, T/T0, rho/rho0    
        ""

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
        """
    
    def solve_flow_field(self, theta_s, theta_c, Vr0, dVr0):
        '''
        Solves the Taylor-Maccoll equation and returns a DataFrame with results.

        Parameters
        ----------
        theta_s : float
            Shock angle (radians).
        theta_c : float
            Cone angle (radians).
        Vr0 : float
            Initial radial velocity.
        dVr0 : float
            Initial derivative of Vr.

        Returns
        -------
        pd.DataFrame
            DataFrame containing Theta (radians), V_r, and V_theta.
        '''
        isentropic_solver = IsentropicRelationsSolver(self.gamma)

        theta = theta_c
        Vr = Vr0
        dVr = dVr0
        M = self.compute_mach(Vr, dVr)

        isentropic_properties = isentropic_solver.isentropic_relations(M)
        p_ratio = isentropic_properties["Static Pressure Ratio (p/p0)"]
        t_ratio = isentropic_properties["Static Temperature Ratio (T/T0)"]
        rho_ratio = isentropic_properties["Static Density Ratio (rho/rho0)"]

        # Lists to store results
        results = [[theta, M, Vr, dVr, p_ratio, t_ratio, rho_ratio]]  # Log initial conditions

        while abs(theta - theta_s) > 1e-3:                            # Continue until reach shock angle
            # Perform RK4 step
            Vr, dVr = self.rk4_step(theta, Vr, dVr)
            M = self.compute_mach(Vr, dVr)

            isentropic_properties = isentropic_solver.isentropic_relations(M)
            p_ratio = isentropic_properties["Static Pressure Ratio (p/p0)"]
            t_ratio = isentropic_properties["Static Temperature Ratio (T/T0)"]
            rho_ratio = isentropic_properties["Static Density Ratio (rho/rho0)"]

            theta += self.h

            # Save current results
            results.append([theta, M, Vr, dVr, p_ratio, t_ratio, rho_ratio])

        # Create and return DataFrame
        results_df = pd.DataFrame(results, columns=["Theta (radians)", "Mach", "V_r", "V_theta", "P/P0", "T/T0", "rho/rho0"])
        return results_df

if __name__ == "__main__":
    solver = TaylorMaccollSolver()
    theta_s = np.radians(solver.theta_s_deg)
    theta_c = theta_s - np.radians(10)

    Mn2, Mn3 = solver.compute_post_shock_mach_components()
    Delta, M2, _ = solver.compute_deflection_and_postshock_mach(Mn2)
    V_r, V_theta = solver.compute_initial_velocity_components(Mn2, M2) #Was using M3 as input should be M2 as it is now
    df = solver.solve_flow_field(theta_s, theta_c, V_r, dVr0=1)        #M2 was dropped from the 4th position.

    print(df.head())
    df.to_csv('data.csv', index=False)

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