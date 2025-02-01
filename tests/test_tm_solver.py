import pytest
import numpy as np
from taylor_maccoll_solver import TaylorMaccollSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return TaylorMaccollSolver(gamma=1.4)

def test_calculate_velocity_components(solver):
    """
    Test the calculation of the post-shock normalized velocity
    and radial / tangential components *right after* the shock.
    """
    # (Vals from OS relations @ M1 = 10 and gamma = 1.4)
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

def test_calculate_Mach_from_components(solver):
    """
    Test the calculation of the Mach number of the flow
    given its normalized velocity components.
    """
    # This is the inverse case of test_calculate_velocity_components()
    V_r = 0.845154249
    V_theta = 0.097590007
    
    M = solver.calculate_Mach_from_components(V_r, V_theta)

    expected_M = 3.61986846
    assert np.isclose(M, expected_M, atol=1e-3)

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
    Test the Taylor-Maccoll solver by asserting the resultant cone angle.
    """
    # Initial conditions
    # Values come from M1 = 10 and gamma = 1.4 @ wave angle of 30deg
    # Recall, theta0 = wedge angle from OS relations
    theta0 = np.radians(23.4132244)  # Initial angle in radians
    Vr0 = 0.845154249        # Initial radial velocity
    dVr0 = 0.097590007       # Initial derivative of Vr ~ V_theta

    # Call the solve function
    theta_c, Vr, dVr = solver.solve(theta0, Vr0, dVr0)
    
    # Assert V_theta ~ 0 (this is true at the cone angle)
    assert np.isclose(dVr, 0, atol=0.01), \
        f"Solver did not return a value of 0 for V_theta."

    theta_c = np.degrees(theta_c) # Convert from rad to deg for easier comparison
    # Expected cone angle
    expected_theta_c = 26.5909011  # Expected cone angle
    expected_Mc = 3.57846955    # Expected Mach at cone angle.
    
    # Assert first and last Theta values
    assert np.isclose(theta_c, expected_theta_c, rtol=0.01), \
        f"Cone angle mismatch: {theta_c} != {expected_theta_c}"
    print(Vr)
    Mc = solver.calculate_Mach_from_components(Vr, dVr)
    assert np.isclose(Mc, expected_Mc, rtol=0.01), \
        f"Mach_c mismatch: {Mc} != {expected_Mc}"