import numpy as np
import pytest
from src.metric_derivative_solver import compute_metric_values, metric_derivative_solver


def _build_simple_grid():
    """
    Small, mildly-warped grid with a smooth synthetic velocity field,
    matching the shape/conventions used in this module's own __main__
    example.
    """
    np.random.seed(0)
    eta_grid = np.linspace(0, 1, 8)
    xi_grid = np.linspace(0, 1, 8)

    x_vals, y_vals = np.meshgrid(xi_grid, eta_grid, indexing='ij')
    x_vals = x_vals + 0.05 * np.random.rand(*x_vals.shape)
    y_vals = y_vals + 0.05 * np.random.rand(*y_vals.shape)

    u_field = np.sin(np.pi * x_vals) * np.cos(np.pi * y_vals)
    v_field = -np.cos(np.pi * x_vals) * np.sin(np.pi * y_vals)

    return eta_grid, xi_grid, x_vals, y_vals, u_field, v_field


def test_compute_metric_values_shape_and_finiteness():
    eta_grid, xi_grid, x_vals, y_vals, u_field, v_field = _build_simple_grid()
    metric_values = compute_metric_values(eta_grid, xi_grid, x_vals, y_vals, u_field, v_field)

    # 6 stacked fields: [v, u, eta_x, eta_y, xi_x, xi_y]
    assert metric_values.shape == (len(eta_grid), len(xi_grid), 6)
    assert np.all(np.isfinite(metric_values))


def test_manual_and_scipy_interpolation_agree():
    """
    Self-consistency check: the hand-written bilinear interpolator
    ('manual') and scipy's RegularGridInterpolator ('scipy') are two
    independent implementations of the same interpolation and should
    agree closely for identical inputs. This does NOT verify the
    underlying physics against Bowcutt's dissertation (that validation is
    still pending) -- it only guards against the two code paths silently
    diverging from each other.
    """
    eta_grid, xi_grid, x_vals, y_vals, u_field, v_field = _build_simple_grid()
    metric_values = compute_metric_values(eta_grid, xi_grid, x_vals, y_vals, u_field, v_field)

    eta0, xi0 = 0.35, 0.4
    i0 = np.searchsorted(eta_grid, eta0)
    j0 = np.searchsorted(xi_grid, xi0)
    x0, y0 = x_vals[j0, i0], y_vals[j0, i0]
    u0, v0 = u_field[j0, i0], v_field[j0, i0]

    result_scipy = metric_derivative_solver(
        v0, u0, x0, y0, eta0, xi0, (eta_grid, xi_grid), metric_values, method='scipy'
    )
    result_manual = metric_derivative_solver(
        v0, u0, x0, y0, eta0, xi0, (eta_grid, xi_grid), metric_values, method='manual'
    )

    for scipy_val, manual_val, name in zip(result_scipy, result_manual, ["eta1", "xi1", "x1", "y1"]):
        assert np.isclose(scipy_val, manual_val, atol=1e-6), (
            f"{name} disagreement between manual and scipy interpolation: "
            f"{manual_val} vs {scipy_val}"
        )


def test_metric_derivative_solver_rejects_unknown_method():
    eta_grid, xi_grid, x_vals, y_vals, u_field, v_field = _build_simple_grid()
    metric_values = compute_metric_values(eta_grid, xi_grid, x_vals, y_vals, u_field, v_field)

    with pytest.raises(ValueError):
        metric_derivative_solver(
            0, 0, 0, 0, 0.5, 0.5, (eta_grid, xi_grid), metric_values, method='bogus'
        )
