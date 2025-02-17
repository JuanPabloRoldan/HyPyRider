import pytest
import numpy as np
from src.isentropic_relations_solver import IsentropicRelationsSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return IsentropicRelationsSolver(gamma=1.4)

def test_isentropic_relation(solver):
    """
    Test the calculation of the isentropic relations solver.
    """
    
    isentropic_relations_dict = solver.isentropic_relations(Mach=5)

    Mach = isentropic_relations_dict["Mach Number"]
    Specific_Heat_Ratio = isentropic_relations_dict["Specific Heat Ratio"]
    Static_Pressure_Ratio = isentropic_relations_dict["Static Pressure Ratio (p/p0)"]
    Static_Temperature_Ratio = isentropic_relations_dict["Static Temperature Ratio (T/T0)"]
    Static_Density_Ratio  = isentropic_relations_dict["Static Density Ratio (rho/rho0)"]
    
    # Expected value
    expected_Mach = 5
    expected_Specific_Heat_Ratio = 1.4
    expected_Static_Pressure_Ratio = 0.00189003
    expected_Static_Temperature_Ratio = 0.16666666
    expected_Static_Density_Ratio = 0.01134023

    assert np.isclose(Mach,expected_Mach, atol=1e-3)
    assert np.isclose(Specific_Heat_Ratio, expected_Specific_Heat_Ratio, atol=1e-3)
    assert np.isclose(Static_Pressure_Ratio, expected_Static_Pressure_Ratio, atol=1e-3)
    assert np.isclose(Static_Temperature_Ratio, expected_Static_Temperature_Ratio, atol=1e-3)
    assert np.isclose(Static_Density_Ratio, expected_Static_Density_Ratio, atol=1e-3)

    
