import numpy as np
import pytest

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
    # (Vals from OS relations @ M1 = 10 and gamma = 1.4)
    # Assume M2, shock angle, flow deflection angle are inputted correctly
    M2 = 3.61986846
    theta_s = np.radians(30)
    delta = np.radians(23.4132244)

    V_prime, V_r, V_theta = solver.calculate_velocity_components(M2, theta_s, delta)

    # Expected values
    expected_V_prime = 0.850769954
    expected_V_r = 0.845154249
    expected_V_theta = -0.097590007
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
    Test the Taylor-Maccoll 2nd order differential equation system (Eqn 2.1).

    expected_result was hand-derived directly from Eqn 2.1: solving the
    equation for d2Vr/dtheta2 gives ddVr = (Vr*dVr**2 - B*C) / (B - dVr**2),
    which requires the Vr*dVr**2 term above -- not dVr**2 alone. The old
    expected value (-1.622309628) was a self-referential golden value from
    a formula missing that Vr factor; it happened to still fall within this
    test's atol=1e-3 by coincidence (Vr=0.8 and dVr**2=1e-4 keep the delta
    small here), which is why it went undetected until checked against the
    paper directly.
    """

    theta = np.radians(25)
    Vr = 0.8
    dVr = 0.01
    result = solver.taylor_maccoll_system(theta, Vr, dVr)

    expected_result = np.array([dVr, -1.6225878698])
    assert np.allclose(result, expected_result, atol=1e-6)

def test_solve(solver):
    """
    Test the Taylor-Maccoll solver by asserting the resultant cone angle.

    solve() integrates from the shock angle toward the cone axis, so
    theta0 here must be theta_s (the shock angle), not the flow deflection
    angle -- passing the deflection angle was the bug this test used to
    (silently) lock in, since the old rtol=0.01 tolerance was loose enough
    to mask the ~0.19deg error it produced.
    """
    # Values come from M1 = 10 and gamma = 1.4 @ wave angle of 30deg
    theta_s = np.radians(30)  # Shock angle, in radians
    Vr0 = 0.845154249        # Radial velocity immediately behind the shock
    dVr0 = -0.097590007       # d(Vr)/dtheta immediately behind the shock ~ V_theta

    # Call the solve function
    theta_c, Vr, dVr = solver.solve(theta_s, Vr0, dVr0)

    # Assert V_theta ~ 0 (this is true at the cone angle)
    assert np.isclose(dVr, 0, atol=0.01), \
        "Solver did not return a value of 0 for V_theta."

    # Expected cone angle, from Bowcutt's own worked example for this case
    # (dissertation p.14, and cited again as the demo values in this
    # module's __main__ block).
    expected_theta_c = np.radians(26.5909011)  # Expected cone angle
    expected_Mc = 3.57846955    # Expected Mach at cone angle.

    # Assert first and last Theta values
    assert np.isclose(theta_c, expected_theta_c, rtol=0.01), \
        f"Cone angle mismatch: {theta_c} != {expected_theta_c}"

    Mc = solver.calculate_Mach_from_components(Vr, dVr)
    assert np.isclose(Mc, expected_Mc, rtol=0.01), \
        f"Mach_c mismatch: {Mc} != {expected_Mc}"

def test_tabulate_from_shock_to_cone(solver):
    theta_s = np.radians(30)
    theta_c = np.radians( 26.5909011)
    Mc = 3.57846955

    V_prime, V_r, V_theta = solver.calculate_velocity_components(Mc, theta_c, theta_c)

    df = solver.tabulate_from_shock_to_cone(theta_s, theta_c, V_r, V_theta)

    tab_cone = df.iloc[0]
    tab_theta_c = tab_cone["Theta (radians)"]
    assert np.isclose(tab_theta_c, theta_c, atol=0.01)

    tab_shock = df.iloc[-1]
    tab_theta_s = tab_shock["Theta (radians)"]
    assert np.isclose(tab_theta_s, theta_s, atol=0.01)
