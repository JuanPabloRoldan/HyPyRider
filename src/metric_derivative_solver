import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

def compute_metric_values(eta_grid, xi_grid, x_vals, y_vals, u_field, v_field):
    deta = eta_grid[1] - eta_grid[0]
    dxi = xi_grid[1] - xi_grid[0]

    dx_deta = np.gradient(x_vals, deta, axis=0)
    dx_dxi = np.gradient(x_vals, dxi, axis=1)
    dy_deta = np.gradient(y_vals, deta, axis=0)
    dy_dxi = np.gradient(y_vals, dxi, axis=1)

    detJ = dx_deta * dy_dxi - dx_dxi * dy_deta
    detJ[detJ == 0] = 1e-12  # avoid divide-by-zero

    eta_x =  dy_dxi / detJ
    eta_y = -dx_dxi / detJ
    xi_x  = -dy_deta / detJ
    xi_y  =  dx_deta / detJ

    metric_values = np.stack([v_field, u_field, eta_x, eta_y, xi_x, xi_y], axis=-1)
    return metric_values

def metric_derivative_solver(v0, u0, x0, y0, eta0, xi0, grid_points, metric_values, method='manual'):
    eta_grid, xi_grid = grid_points

    def bilinear_interpolate(eta, xi):
        i = np.searchsorted(eta_grid, eta) - 1
        j = np.searchsorted(xi_grid, xi) - 1

        i = np.clip(i, 0, len(eta_grid) - 2)
        j = np.clip(j, 0, len(xi_grid) - 2)

        eta1, eta2 = eta_grid[i], eta_grid[i + 1]
        xi1, xi2 = xi_grid[j], xi_grid[j + 1]

        t = (eta - eta1) / (eta2 - eta1)
        s = (xi - xi1) / (xi2 - xi1)

        Q11 = metric_values[i, j]
        Q21 = metric_values[i + 1, j]
        Q12 = metric_values[i, j + 1]
        Q22 = metric_values[i + 1, j + 1]

        interp = (
            Q11 * (1 - t) * (1 - s) +
            Q21 * t * (1 - s) +
            Q12 * (1 - t) * s +
            Q22 * t * s
        )

        return interp

    if method == 'scipy':
        interpolator = RegularGridInterpolator((eta_grid, xi_grid), metric_values)

        def get_metric(eta, xi):
            return interpolator([[eta, xi]])[0]

    elif method == 'manual':
        def get_metric(eta, xi):
            return bilinear_interpolate(eta, xi)

    else:
        raise ValueError("Invalid interpolation method. Use 'scipy' or 'manual'.")

    def odes(t, z):
        x, y, eta, xi = z
        v, u, eta_x, eta_y, xi_x, xi_y = get_metric(eta, xi)

        dxdt = u
        dydt = v
        detadt = eta_x * u + eta_y * v
        dxidt = xi_x * u + xi_y * v

        return [dxdt, dydt, detadt, dxidt]

    z0 = [x0, y0, eta0, xi0]
    t_span = (0, 1)
    sol = solve_ivp(odes, t_span, z0)
    x1, y1, eta1, xi1 = sol.y[:, -1]

    return eta1, xi1, x1, y1

# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Create grid
    eta_grid = np.linspace(0, 1, 10) # Change this grid size as needed
    xi_grid = np.linspace(0, 1, 10) # Change this grid size as needed

    # Meshgrid for x/y positions (simulate body-fitted grid)
    x_vals, y_vals = np.meshgrid(xi_grid, eta_grid, indexing='ij')

    # Add perturbation to simulate warping
    x_vals += 0.05 * np.random.rand(*x_vals.shape)
    y_vals += 0.05 * np.random.rand(*y_vals.shape)

    # Generate random but smooth velocity fields
    u_field = np.sin(np.pi * x_vals) * np.cos(np.pi * y_vals)
    v_field = -np.cos(np.pi * x_vals) * np.sin(np.pi * y_vals)

    # Compute full metric values
    metric_values = compute_metric_values(eta_grid, xi_grid, x_vals, y_vals, u_field, v_field)

    # Starting point
    eta0, xi0 = 0.15, 0.15 # Change this to your desired starting point
    i0 = np.searchsorted(eta_grid, eta0)
    j0 = np.searchsorted(xi_grid, xi0)
    x0 = x_vals[j0, i0]
    y0 = y_vals[j0, i0]
    u0 = u_field[j0, i0]
    v0 = v_field[j0, i0]

    # Run both methods
    result_scipy = metric_derivative_solver(v0, u0, x0, y0, eta0, xi0,
                                            (eta_grid, xi_grid), metric_values, method='scipy')

    result_manual = metric_derivative_solver(v0, u0, x0, y0, eta0, xi0,
                                             (eta_grid, xi_grid), metric_values, method='manual')

    print("Scipy Interpolation Result:")
    print(f"  eta1 = {result_scipy[0]:.8f}, xi1 = {result_scipy[1]:.8f}, x1 = {result_scipy[2]:.8f}, y1 = {result_scipy[3]:.8f}")

    print("\n Manual Interpolation Result:")
    print(f"  eta1 = {result_manual[0]:.8f}, xi1 = {result_manual[1]:.8f}, x1 = {result_manual[2]:.8f}, y1 = {result_manual[3]:.8f}")

# Test