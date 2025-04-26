import os
import numpy as np
from stl.mesh import Mesh
import pyvista as pv
from streamline_integrator import StreamlineIntegrator
from flow_interpolator import bilinear_interpolate_moc
from axi_sym_MoC_solver import MoC_Skeleton
from moc_to_structured_grid import extract_fields_from_moc, build_logical_coords, build_inverse_coordinate_map
from scipy.interpolate import NearestNDInterpolator

class ExpansionSurfacePressureSolver:
    def __init__(self, stl_file, gamma=1.2, M1=7.0, theta_s=np.radians(20)):
        """
        Initializes the expansion surface solver for a given STL mesh.
        """
        self.gamma = gamma
        self.M1 = M1
        self.theta_s = theta_s
        self.mesh = Mesh.from_file(stl_file)
        self.surface_points = self.mesh.vectors.reshape(-1, 3)

        # Set up the interpolator from the streamline solution
        self.integrator = StreamlineIntegrator(gamma, M1, theta_s, surface="expansion")

        # Load MOC-generated mesh
        T = 231.64  # K
        Rgas = 287  # J/kg/K
        a_inf = np.sqrt(gamma * Rgas * T)
        wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}
        moc = MoC_Skeleton(M1, a_inf, gamma, wall_params)
        moc_mesh = moc.MoC_Mesher()
        self.x_vals, self.r_vals, self.u_field, self.v_field = extract_fields_from_moc(moc_mesh)

        # Interpolation mapping
        eta_grid, xi_grid, eta_vals, xi_vals = build_logical_coords(moc_mesh)
        eta_lin, xi_lin = build_inverse_coordinate_map(self.x_vals, self.r_vals, eta_vals, xi_vals)
        mask = ~np.isnan(self.x_vals) & ~np.isnan(self.r_vals)
        pts_map = np.stack((self.x_vals[mask], self.r_vals[mask]), axis=-1)

        self.eta_interp = lambda x, r: float(eta_lin(x, r)) if not np.isnan(eta_lin(x, r)) else float(NearestNDInterpolator(pts_map, eta_vals[mask])(x, r))
        self.xi_interp = lambda x, r: float(xi_lin(x, r)) if not np.isnan(xi_lin(x, r)) else float(NearestNDInterpolator(pts_map, xi_vals[mask])(x, r))

    def get_surface_r(self, y, z):
        """
        Compute radial distance r from y and z (shifted properly).
        """
        y_shifted = y + 4.55  # undo y-shift
        return np.hypot(y_shifted, z)

    def compute_local_flow(self, x, r):
        """
        Get (u,v) at (x,r) using bilinear interpolation on MOC expansion field.
        """
        eta = self.eta_interp(x, r)
        xi = self.xi_interp(x, r)
        uv = bilinear_interpolate_moc(eta, xi, {"u_field": self.u_field, "v_field": self.v_field})
        if uv is None or np.isnan(uv['u']) or np.isnan(uv['v']):
            return 0.0, 0.0
        return uv['u'], uv['v']

    def compute_surface_pressure(self):
        """
        Loop through the surface points and compute pressure-related quantities.
        """
        self.surface_results = []

        # Define freestream conditions
        T_inf = 231.64  # Freestream temperature (K)
        Rgas = 287.0    # Gas constant (J/kg/K)
        gamma = self.gamma
        M_inf = self.M1

        a_inf = np.sqrt(gamma * Rgas * T_inf)  # Freestream speed of sound
        V_inf = M_inf * a_inf                  # Freestream velocity
        p_inf = 1200.0                         # Freestream static pressure (Pa) - adjust if needed
        rho_inf = p_inf / (Rgas * T_inf)        # Freestream density
        q_inf = 0.5 * rho_inf * V_inf**2        # Dynamic pressure

        # Freestream stagnation pressure
        p0_inf = p_inf * (1 + (gamma-1)/2 * M_inf**2)**(gamma/(gamma-1))

        for pt in self.surface_points:
            x, y, z = pt
            r = self.get_surface_r(y, z)

            # Find local velocity components (u,v)
            u, v = self.compute_local_flow(x, r)

            # === Robustness Guard ===
            if np.isnan(u) or np.isnan(v):
                continue  # Skip if interpolation failed

            # Find local Mach number
            V_mag = np.hypot(u, v)
            Mach_local = V_mag / a_inf

            # Local static pressure (isentropic expansion)
            p_local = p0_inf * (1 + (gamma-1)/2 * Mach_local**2)**(-gamma/(gamma-1))

            # Compute Cp correctly normalized
            Cp_local = (p_local - p_inf) / q_inf

            self.surface_results.append((x, y, z, Cp_local))



    def export_to_vtk(self, output_file="src/outputs/expansion_surface_solution2.vtk"):
        """
        Export the computed surface pressure coefficients to a VTK file.
        """
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        points = self.surface_points
        faces = np.hstack([np.full((len(self.mesh.vectors), 1), 3), 
                        np.arange(len(points)).reshape(-1, 3)]).astype(np.int64)

        pv_mesh = pv.PolyData(points, faces)

        # Compute Cp for each triangle (cell)
        Cp_per_point = np.array([res[3] for res in self.surface_results])

        # Now average Cp for each triangle (3 vertices per face)
        n_faces = self.mesh.vectors.shape[0]
        Cp_cells = []
        for i in range(n_faces):
            i0, i1, i2 = i*3, i*3+1, i*3+2
            Cp_avg = (Cp_per_point[i0] + Cp_per_point[i1] + Cp_per_point[i2]) / 3
            Cp_cells.append(Cp_avg)
        
        Cp_cells = np.array(Cp_cells)

        pv_mesh.cell_data["Cp Expansion"] = Cp_cells  # assign per cell

        pv_mesh.save(output_file)
        print(f"Expansion surface solution saved to: {output_file}")

if __name__ == "__main__":
    solver = ExpansionSurfacePressureSolver(stl_file="src/inputs/expansionSurfaceMeshed.stl")
    solver.compute_surface_pressure()
    solver.export_to_vtk()
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract x and Cp values
    x_vals = np.array([pt[0] for pt in solver.surface_results])
    Cp_vals = np.array([pt[3] for pt in solver.surface_results])

    # Optionally, sort by x (not strictly necessary if your mesh is organized)
    sort_idx = np.argsort(x_vals)
    x_vals = x_vals[sort_idx]
    Cp_vals = Cp_vals[sort_idx]

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(x_vals, Cp_vals, marker='o', linestyle='-', markersize=3)
    plt.xlabel('x-location (m)')
    plt.ylabel('Pressure Coefficient Cp')
    plt.title('Cp Distribution Along Expansion Surface')
    plt.grid(True)
    plt.show()

