import unittest
from HyPyRider.CompressionSurfaceDesign.inputs.velocity_altitude_map import (
    calculate_pressure, calculate_temperature, calculate_density, calculate_dynamic_pressure
)

class TestVelocityAltitudeMap(unittest.TestCase):
    def test_calculate_pressure(self):
        altitude = 10000  # 10 km
        pressure = calculate_pressure(altitude)
        print(pressure)

    def test_calculate_temperature(self):
        altitude = 10000  # 10 km
        temperature = calculate_temperature(altitude)
        print(temperature)

    def test_calculate_density(self):
        pressure = 101325  # Sea level pressure
        temperature = 15.04  # Sea level temperature (Celsius)
        density = calculate_density(pressure, temperature)
        print(density)

    def test_calculate_dynamic_pressure(self):
        gamma = 1.4
        pressure = 101325
        mach = 2.0
        dynamic_pressure = calculate_dynamic_pressure(gamma, pressure, mach)
        print(dynamic_pressure)

if __name__ == "__main__":
    unittest.main()