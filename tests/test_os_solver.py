import pytest
import numpy as np
from src.oblique_shock_solver import ObliqueShockSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return ObliqueShockSolver(gamma=1.4)

def test_calculate_post_shock_mach_and_deflection(solver):
    """
    Test the calculation of the post-shock Mach number (M2) and flow deflection angle (delta).
    """
    M1 = 10.0
    theta_s = np.radians(30)
    delta, M2 = solver.calculate_post_shock_mach_and_deflection(M1, theta_s)
    
    # Expected value
    expected_delta = np.radians(23.4132244)
    expected_M2 = 3.61986846
    assert np.isclose(delta, expected_delta, atol=1e-3)
    assert np.isclose(M2, expected_M2, atol=1e-3)

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
    assert np.isclose(result["delta"], expected_delta, atol=1e-3)
    assert np.isclose(result["M2"], expected_M2, atol=1e-3)