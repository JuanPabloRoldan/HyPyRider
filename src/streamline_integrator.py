import os
import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer
import process_LE_points
from flow_interpolator import bilinear_interpolate_moc
from axi_sym_MoC_solver import MoC_Skeleton
from moc_to_structured_grid import extract_fields_from_moc, build_logical_coords, build_inverse_coordinate_map
from scipy.interpolate import NearestNDInterpolator

class StreamlineIntegrator:
    def __init__(self, gamma, M1, theta_s, surface="compression"):
        """
        Initializes the streamline integrator.
        """
        self.gamma = gamma
        self.M1 = M1
        self.theta_s = theta_s
        self.surface = surface
        self.streamline_data = []  # store (x, r, z, id, order)

        if surface == "expansion":
            # set up MOC mesh and interpolation
            T = 231.64
            Rgas = 287
            a_inf = np.sqrt(gamma * Rgas * T)
            wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}
            moc = MoC_Skeleton(M1, a_inf, gamma, wall_params)
            moc_mesh = moc.MoC_Mesher()
            x_vals, r_vals, u_field, v_field = extract_fields_from_moc(moc_mesh)

            # logical <-> physical mapping
            eta_grid, xi_grid, eta_vals, xi_vals = build_logical_coords(moc_mesh)
            eta_lin, xi_lin = build_inverse_coordinate_map(x_vals, r_vals, eta_vals, xi_vals)
            mask = ~np.isnan(x_vals) & ~np.isnan(r_vals)
            pts_map = np.stack((x_vals[mask], r_vals[mask]), axis=-1)
            eta_near = NearestNDInterpolator(pts_map, eta_vals[mask])
            xi_near = NearestNDInterpolator(pts_map, xi_vals[mask])
            self.eta_interp = lambda x, r: float(eta_lin(x, r)) if not np.isnan(eta_lin(x, r)) else float(eta_near(x, r))
            self.xi_interp = lambda x, r: float(xi_lin(x, r)) if not np.isnan(xi_lin(x, r)) else float(xi_near(x, r))

            # velocity fallback
            pts_vel = pts_map
            self.u_near = NearestNDInterpolator(pts_vel, u_field[mask])
            self.v_near = NearestNDInterpolator(pts_vel, v_field[mask])
            self.flow_fields = {"u_field": u_field, "v_field": v_field}

        if surface == "compression":
            self.conical_analyzer = ConicalFlowAnalyzer(M1, gamma)
            self.theta_c, _, _ = self.conical_analyzer.solve_taylor_maccoll(theta_s)
            self.TM_tabulation = self.conical_analyzer.tabulate_tm_shock_to_cone(theta_s)

    def get_flow(self, x, r, theta=None):
        """
        Fetch axial (u) and radial (v) velocities at physical (x,r).
        """
        if self.surface == "compression":
            u = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_r'])
            v = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_theta'])
            return u, v

        # expansion surface lookup
        eta = self.eta_interp(x, r)
        xi = self.xi_interp(x, r)
        uv = bilinear_interpolate_moc(eta, xi, self.flow_fields)
        if uv is None or np.isnan(uv['u']) or np.isnan(uv['v']):
            u = self.u_near(x, r)
            v = self.v_near(x, r)
        else:
            u, v = uv['u'], uv['v']
        return u, v

    def trace_streamline(self, x_le, y_le, z_le, streamline_id):
        """
        March a single streamline in 3D from the expansion leading edge.
        x_le,y_le,z_le are normalized coordinates of the LE.
        """
        # convert to physical
        x = x_le * self.ref_length
        # initial radial angle and distance
        phi0 = np.arctan2(z_le * self.ref_length, y_le * self.ref_length)
        r = np.hypot(y_le * self.ref_length, z_le * self.ref_length)
        dt = 1e-4
        points = []

        # starting point in (x,y,z)
        y = r * np.cos(phi0)
        z = r * np.sin(phi0)
        points.append([x, y, z, streamline_id, 0])

        # march until end
        for order in range(1, 50000):
            u, v = self.get_flow(x, r)
            if u is None or np.isnan(u):
                print(f"[Streamline {streamline_id}] interp failed @ x={x:.6f}, r={r:.6f}")
                break
            x += u * dt
            r += v * dt
            y = r * np.cos(phi0)
            z = r * np.sin(phi0)
            points.append([x, y, z, streamline_id, order])
            if x >= self.ref_length:
                break

        self.streamline_data.extend(points)

    def create_lower_surface(self):
        """
        Trace streamlines from each leading-edge point.
        """
        self.streamline_data = []
        self.ref_length = self.LE_points['X'].max()
        for idx, row in self.LE_points.iterrows():
            x_le = row['X'] / self.ref_length
            y_le = row['Y'] / self.ref_length
            z_le = row['Z'] / self.ref_length
            self.trace_streamline(x_le, y_le, z_le, idx)

    def export_streamlines_dat(self, filename):
        """
        Save all traced streamlines to a .dat file.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for sid in sorted({p[3] for p in self.streamline_data}):
                seg = [p for p in self.streamline_data if p[3] == sid]
                f.write(f"{len(seg)}\n")
                for x, y, z, _, _ in seg:
                    f.write(f"{x}\t{y-4.55}\t{z}\n")

    def close_streamline_segments(self, filename):
        """
        Optionally link adjacent streamline segments.
        """
        pass
