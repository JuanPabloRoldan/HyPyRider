import pytest
import numpy as np
from src.taylor_maccoll_solver import TaylorMaccollSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return TaylorMaccollSolver(gamma=1.4)

def test_calculate_velocity_components(solver):
    """
    Test the calculation of the post-shock normalized velocity
    and radial / tangential components *right after* the shock.
    """
    
    # Assume M2, shock angle, flow deflection angle are inputted correctly
    M2 = 3.61986846
    theta_s = np.radians(30)
    delta = np.radians(23.4132244)

    V_prime, V_r, V_theta = solver.calculate_velocity_components(M2, theta_s, delta)

    # Expected values
    expected_V_prime = 0.850769954
    expected_V_r = 0.845154249
    expected_V_theta = 0.097590007
    assert np.isclose(V_prime, expected_V_prime, atol=1e-3)
    assert np.isclose(V_r, expected_V_r, atol=1e-3)
    assert np.isclose(V_theta, expected_V_theta, atol=1e-3)

def test_taylor_maccoll_system(solver):
    """
    Test the Taylor-Maccoll 2nd order differential equation system
    """

    theta = np.radians(25)
    Vr = 0.8
    dVr = 0.01
    result = solver.taylor_maccoll_system(theta, Vr, dVr)
    
    expected_result = np.array([dVr, -1.622309628])
    assert np.allclose(result, expected_result, atol=1e-3)

def test_solve(solver):
    """
    Test the Taylor-Maccoll solver by asserting the first (shock) and last (cone) values of theta.
    """
    # Initial conditions
    theta0 = np.radians(30)  # Initial angle in radians
    Vr0 = 0.845154249        # Initial radial velocity
    dVr0 = 0.097590007       # Initial derivative of Vr ~ V_theta

    # Call the solve function
    results = solver.solve(theta0, Vr0, dVr0)

    # Validate the output structure
    assert "Theta (degrees)" in results.columns
    assert "V_r" in results.columns
    assert "V_theta" in results.columns

    # Expected first and last Theta values (replace with known benchmarks if available)
    expected_first_theta = 30         # First theta should always match theta0 in degrees
    expected_last_theta = 26.5909011  # Replace with expected value for the test case
    
    # Assert first and last Theta values
    assert np.isclose(results["Theta (degrees)"].iloc[0], expected_first_theta, atol=1e-3), \
        f"First Theta mismatch: {results['Theta (degrees)'].iloc[0]} != {expected_first_theta}"
    assert np.isclose(results["Theta (degrees)"].iloc[-1], expected_last_theta, atol=1e-3), \
        f"Last Theta mismatch: {results['Theta (degrees)'].iloc[-1]} != {expected_last_theta}"