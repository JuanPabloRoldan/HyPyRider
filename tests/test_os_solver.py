import pytest
import numpy as np
from src.oblique_shock_solver import ObliqueShockSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return ObliqueShockSolver(gamma=1.4)

def test_calculate_post_shock_conditions(solver):
    """
    Test the calculation of all post-shock conditions.
    TODO: add pressure, temperature, and density ratios
    """
    M1 = 2.0
    theta_s = np.radians(50)
    result = solver.calculate_post_shock_conditions(M1, theta_s)

    # Expected values
    expected_delta = np.radians(18.1299559)
    expected_M2 = 1.30688201
    expected_P2_P1 = 2.57184574
    expected_rho2_rho1 = 1.91686539
    expected_T2_T1 = 1.34169345

    assert np.isclose(result["delta"], expected_delta, atol=1e-3), \
        f"Turn angle mismatch: {result['delta']} != {expected_delta}"
    assert np.isclose(result["M2"], expected_M2, atol=1e-3), \
        f"M2 mismatch: {result['M2']} !=  {expected_M2}"
    assert np.isclose(result["P2_P1"], expected_P2_P1, atol=1e-3), \
        f"Static pressure ratio mismatch: {result['P2_P1']} !=  {expected_P2_P1}"
    assert np.isclose(result["rho2_rho1"], expected_rho2_rho1, atol=1e-3), \
        f"Density ratio mismatch: {result['rho2_rho1']} != {expected_rho2_rho1}"
    assert np.isclose(result["T2_T1"], expected_T2_T1, atol=1e-3), \
        f"Static temperature ratio mismatch: {result['T2_T1']} != {expected_T2_T1}"