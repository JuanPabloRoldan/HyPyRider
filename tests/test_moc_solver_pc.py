import numpy as np
import pytest
from src.moc_solver_pc import AxisymMoC
from src.point import Point

# Wall/flow parameters matching axi-sym_MoC_solver.py's own __main__ example
# (M_inf=7, gamma=1.2, at 30,000 m). q_max computed the same way MoC_Skeleton
# computes it, so these fixture points are self-consistent with a real mesh.
GAMMA = 1.2
Q_MAX = 2169.5234831639873
WALL_PARAMS = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}


@pytest.fixture
def solver():
    return AxisymMoC(q_max=Q_MAX, gamma=GAMMA, wall_params=WALL_PARAMS)


def test_solve_internal_point_matches_known_mesh_point(solver):
    """
    Regression test: PA/PB/expected-PC below were captured from an actual
    run of MoC_Skeleton.MoC_Mesher() with the parameters above (points
    mesh[9][5], mesh[10][4], mesh[10][5]). This locks in current behavior
    as a change-detector for solve_internal_point -- it is NOT an
    independent verification against Bowcutt's dissertation, which is
    still pending.
    """
    PA = Point(x=5.148697413726159, r=3.54749733765658, theta=-0.06749373622574202,
               M=7.41593070819111, q=1995.660259749471)
    PB = Point(x=5.111178875710549, r=3.5896159582877574, theta=-0.05229025710246242,
               M=7.3195636557447825, q=1991.603615559419)

    PC = solver.solve_internal_point(PA, PB)

    assert np.isclose(PC.x, 5.278803348157691, rtol=1e-6)
    assert np.isclose(PC.r, 3.556342326626543, rtol=1e-6)
    assert np.isclose(PC.theta, -0.06734593291497436, rtol=1e-5, atol=1e-8)
    assert np.isclose(PC.M, 7.414713702532491, rtol=1e-6)
    assert np.isclose(PC.q, 1995.6098612984088, rtol=1e-6)


def test_solve_wall_point_succeeds_when_intersection_exists(solver):
    """
    Regression test using a real point from a successful mesh row (see
    test_solve_internal_point_matches_known_mesh_point for provenance
    notes). Note this PB's x (10.08) is already past wall_params["x2"]
    (9.39) -- the predictor line still happens to intersect the
    extrapolated parabola here, it just isn't guaranteed to in general
    (see the next test).
    """
    PB = Point(x=10.084175290859454, r=2.3015659141074742, theta=-0.3524386403089323,
               M=9.437775698916417, q=2057.118908975743)

    PC = solver.solve_wall_point(PB)

    assert PC is not None
    assert np.isfinite(PC.x) and np.isfinite(PC.r)


def test_solve_wall_point_returns_none_past_wall_domain(solver, capsys):
    """
    Investigation finding: running the full 50x50 mesh from
    axi-sym_MoC_solver.py's own example, solve_wall_point() first fails
    (discriminant < 0, "no real intersection") at mesh row 18, where PB.x
    = 11.27 -- about 20% beyond wall_params["x2"] = 9.39, the parabola's
    defined z-extent. The intersection math itself checks out algebraically
    (it is the standard line/parabola quadratic); the failure is a
    consequence of the mesh outrunning the geometry it was given, not a
    bug in the solve. This test locks in that observed behavior and checks
    that the failure is now reported with a specific reason instead of a
    bare "neg".
    """
    PB = Point(x=11.274183599427833, r=1.806742556073436, theta=-0.4059936956119819,
               M=9.779187079764256, q=2064.278696921383)

    PC = solver.solve_wall_point(PB)

    assert PC is None
    assert "outside the wall's defined domain" in capsys.readouterr().out
