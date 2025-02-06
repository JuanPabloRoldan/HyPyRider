import pytest
import numpy as np
from src.isentropic_relations_solver import IsentropicRelationsSolver

# Fixture to initialize the solver instance
@pytest.fixture
def solver():
    return IsentropicRelationsSolver(mach=5,gamma=1.4)

def test_isentropic_relation(solver):
    """
    Test the calculation of the post-shock Mach number (M2) and flow deflection angle (delta).
    """
    
   isentropic_relations_dict = solver.isentropic_relations()

   Mach = isentropic_relations_dict["Mach Number"]
   Specific_Heat_Ratio = isentropic_relations_dict["Specific Heat Ratio"]
   Static_Pressure_Ratio = isentropic_relations_dict["Static Pressure Ratio (p/p0)"]
   Static_Temperature_Ratio = isentropic_relations_dict["Static Temperature Ratio (T/T0)"]
   Static_Density_Ratio  = isentropic_relations_dict["Static Density Ratio (rho/rho0)"]
   Total_Pressure_Ratio = isentropic_relations_dict["Total Pressure Ratio (p0/p0*)"]
   Total_Temperature_Ratio = isentropic_relations_dict["Total Temperature Ratio (T0/T0*)"]
   Speed_of_Sound_Ratio = isentropic_relations_dict["Speed of Sound Ratio (a/a*)"]
    
    # Expected value
    expected_Mach = 5
    expected_Specific_Heat_Ratio = 1.4
    expected_Static_Pressure_Ratio = 0.00189003
    expected_Static_Temperature_Ratio = 0.16666666
    expected_Static_Density_Ratio = 0.01134023
    expected_Total_Pressure_Ratio = 0.00357770
    expected_Total_Temperature_Ratio = 0.2
    expected_Speed_of_Sound_Ratio = 25.0000000

    assert np.isclose(Mach,expected_Mach, atol=1e-3)
    assert np.isclose(Specific_Heat_Ratio, expected_Specific_Heat_Ratio, atol=1e-3)
    assert np.isclose(Static_Pressure_Ratio, expected_Static_Pressure_Ratio, atol=1e-3)
    assert np.isclose(Static_Temperature_Ratio, expected_Static_Temperature_Ratio, atol=1e-3)
    assert np.isclose(Static_Density_Ratio, expected_Static_Density_Ratio, atol=1e-3)
    assert np.isclose(Total_Pressure_Ratio, expected_Total_Pressure_Ratio, atol=1e-3)
    assert np.isclose(Total_Temperature_Ratio, expected_Total_Temperature_Ratio, atol=1e-3)
    assert np.isclose(Speed_of_Sound_Ratio, expected_Speed_of_Sound_Ratio, atol=1e-3)

    
