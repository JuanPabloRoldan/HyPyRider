import numpy as np
from scipy.interpolate import LinearNDInterpolator
from axi_sym_MoC_solver import MoC_Skeleton
import pandas as pd

def extract_fields_from_moc(moc_mesh):
    i_max, j_max = moc_mesh.shape

    x_vals = np.full((i_max, j_max), np.nan)
    r_vals = np.full((i_max, j_max), np.nan)
    u_field = np.full((i_max, j_max), np.nan)
    v_field = np.full((i_max, j_max), np.nan)

    for i in range(i_max):
        for j in range(j_max):
            pt = moc_mesh[i][j]
            if pt is not None:
                x_vals[i, j] = pt.x
                r_vals[i, j] = pt.r
                u_field[i, j] = pt.q * np.cos(pt.theta)
                v_field[i, j] = pt.q * np.sin(pt.theta)

    return x_vals, r_vals, u_field, v_field

def build_logical_coords(moc_mesh):
    eta_grid = np.arange(moc_mesh.shape[0])
    xi_grid  = np.arange(moc_mesh.shape[1])
    eta_vals, xi_vals = np.meshgrid(eta_grid, xi_grid, indexing="ij")
    return eta_grid, xi_grid, eta_vals, xi_vals

def build_inverse_coordinate_map(x_vals, r_vals, eta_vals, xi_vals):
    mask = ~np.isnan(x_vals) & ~np.isnan(r_vals)
    x_flat = x_vals[mask]
    r_flat = r_vals[mask]
    eta_flat = eta_vals[mask]
    xi_flat = xi_vals[mask]

    eta_interp = LinearNDInterpolator(np.stack((x_flat, r_flat), axis=-1), eta_flat)
    xi_interp  = LinearNDInterpolator(np.stack((x_flat, r_flat), axis=-1), xi_flat)

    return eta_interp, xi_interp

def tabulate_moc_flow(x_vals, r_vals, u_field, v_field):
    """
    Converts MoC mesh flow fields into a flat tabulation of (x, r, V_r, V_theta).
    """
    rows = []
    for i in range(x_vals.shape[0]):
        for j in range(x_vals.shape[1]):
            x = x_vals[i, j]
            r = r_vals[i, j]
            u = u_field[i, j]
            v = v_field[i, j]

            if np.isnan(x) or np.isnan(r):
                continue

            V_r = u * (x / np.sqrt(x**2 + r**2)) + v * (r / np.sqrt(x**2 + r**2))
            V_theta = -u * (r / np.sqrt(x**2 + r**2)) + v * (x / np.sqrt(x**2 + r**2))
            
            rows.append((x, r, V_r, V_theta))

    df = pd.DataFrame(rows, columns=["x", "r", "V_r", "V_theta"])
    return df

if __name__ == "__main__":
    # === STEP 1: Recreate your MoC mesh ===
    M_inf = 7
    gamma = 1.2
    a_inf = np.sqrt(gamma * 287 * 231.64) # 30km temperature

    wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}
    moc_solver = MoC_Skeleton(M_inf, a_inf, gamma, wall_params)
    moc_mesh = moc_solver.MoC_Mesher()

    # === STEP 2: Extract structured arrays ===
    x_vals, r_vals, u_field, v_field = extract_fields_from_moc(moc_mesh)
    eta_grid, xi_grid, eta_vals, xi_vals = build_logical_coords(moc_mesh)

    # === STEP 3: Build inverse transform ===
    eta_interp_func, xi_interp_func = build_inverse_coordinate_map(x_vals, r_vals, eta_vals, xi_vals)

    # Example usage:
    x0, r0 = 4.2, 3.6  # Choose a physical point inside your domain
    eta0 = eta_interp_func(x0, r0)
    xi0 = xi_interp_func(x0, r0)

    print(f"(x0, r0) = ({x0}, {r0}) maps to (eta0, xi0) = ({eta0}, {xi0})")

    # === Visualize the MoC grid ===
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    mask = ~np.isnan(x_vals) & ~np.isnan(r_vals)
    plt.scatter(x_vals[mask], r_vals[mask], c='blue', s=10, label='MoC Grid Points')
    plt.scatter([x0], [r0], c='red', label='Test Point')
    plt.xlabel('x')
    plt.ylabel('r')
    plt.title('MoC Domain Coverage')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # === Plot eta-xi grid and test point ===
    plt.figure(figsize=(7, 6))
    plt.title("Logical (eta, xi) Grid")
    plt.xlabel("xi")
    plt.ylabel("eta")

    # Show grid lines
    for eta_line in eta_grid:
        plt.plot(xi_grid, [eta_line]*len(xi_grid), color="gray", linewidth=0.5, alpha=0.5)
    for xi_line in xi_grid:
        plt.plot([xi_line]*len(eta_grid), eta_grid, color="gray", linewidth=0.5, alpha=0.5)

    # Plot test point
    plt.scatter([xi0], [eta0], c='red', s=80, label="Mapped Point (eta, xi)")
    plt.grid(True)
    plt.legend()
    plt.show()