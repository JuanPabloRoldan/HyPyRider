import numpy as np
from src.point import Point


def test_point_stores_fields_and_computes_mach_angle():
    p = Point(x=1.0, r=2.0, theta=np.radians(10), M=2.0, q=0.9)

    assert p.x == 1.0
    assert p.r == 2.0
    assert np.isclose(p.theta, np.radians(10))
    assert p.M == 2.0
    assert p.q == 0.9

    # Mach angle: mu = asin(1/M)
    expected_mu = np.arcsin(0.5)
    assert np.isclose(p.mu, expected_mu, atol=1e-9)


def test_point_mach_angle_at_sonic_is_ninety_degrees():
    # At M=1 (sonic), mu = asin(1/1) = 90 degrees
    p = Point(x=0, r=0, theta=0, M=1.0, q=1.0)
    assert np.isclose(p.mu, np.radians(90), atol=1e-9)


def test_point_repr_reports_degrees_not_radians():
    # __repr__ converts theta/mu to degrees for display; regression guard
    # against that conversion silently disappearing.
    p = Point(x=1.0, r=2.0, theta=np.radians(30), M=2.0, q=0.9)
    text = repr(p)
    assert "theta=30.00" in text
