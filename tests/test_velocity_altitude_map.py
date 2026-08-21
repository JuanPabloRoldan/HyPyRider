import numpy as np
from src.velocity_altitude_map import (
    calculate_pressure,
    calculate_temperature,
    calculate_density,
    calculate_dynamic_pressure,
)


def test_calculate_pressure_sea_level_matches_isa():
    # Standard sea-level pressure is ~101325 Pa; this segmented model's own
    # constants give a value within 1% of that ISA reference.
    p = calculate_pressure(0)
    assert np.isclose(p, 101325, rtol=0.01)


def test_calculate_pressure_returns_pascals_not_kilopascals():
    # calculate_pressure() explicitly converts kPa -> Pa; sea-level pressure
    # in Pa is ~1e5, not ~1e2.
    assert calculate_pressure(0) > 50_000


def test_calculate_temperature_sea_level():
    assert np.isclose(calculate_temperature(0), 15.04, atol=1e-6)


def test_calculate_temperature_stratosphere_is_isothermal():
    # Lower stratosphere (11 km - 25 km) is modeled as isothermal.
    assert calculate_temperature(15000) == -56.46
    assert calculate_temperature(20000) == -56.46


def test_calculate_density_sea_level_matches_isa():
    """
    Regression test for a units bug: calculate_density() previously divided
    Pascal-scale pressure by a gas constant scaled for kPa (0.2869),
    producing a density ~1000x too high (~1225 kg/m^3 instead of the
    correct ~1.225 kg/m^3 at sea level).
    """
    p = calculate_pressure(0)
    t = calculate_temperature(0)
    rho = calculate_density(p, t)
    assert np.isclose(rho, 1.225, rtol=0.01)


def test_calculate_dynamic_pressure_formula():
    # q = 0.5 * gamma * p * M^2
    q = calculate_dynamic_pressure(gamma=1.4, p=101325, M=1.0)
    assert np.isclose(q, 0.5 * 1.4 * 101325, rtol=1e-9)


def test_calculate_dynamic_pressure_scales_with_mach_squared():
    q1 = calculate_dynamic_pressure(gamma=1.4, p=101325, M=1.0)
    q2 = calculate_dynamic_pressure(gamma=1.4, p=101325, M=2.0)
    assert np.isclose(q2, 4 * q1, rtol=1e-9)
