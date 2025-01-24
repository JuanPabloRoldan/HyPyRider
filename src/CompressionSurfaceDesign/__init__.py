"""
CompressionSurfaceDesign package:
Tools for analyzing aerodynamic surfaces and flow dynamics.
"""

from .inputs.process_LE_points import extract_points_from_file
from .inputs.streamline_integrator import StreamlineIntegrator
from .inputs.velocity_altitude_map import calculate_pressure, calculate_temperature, calculate_density, calculate_dynamic_pressure

__all__ = [
    "extract_points_from_file",
    "StreamlineIntegrator",
    "calculate_pressure",
    "calculate_temperature",
    "calculate_density",
    "calculate_dynamic_pressure",
]