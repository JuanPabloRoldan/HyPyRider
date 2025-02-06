import math
class IsentropicRelationsSolver:
  def isentropic_relations(mach, gamma):
    """
    Calculates isentropic relations for given Mach number and specific heat ratio.
  
    Args:
      mach: Mach number.
      gamma: Specific heat ratio.
  
    Returns:
      A dictionary containing the isentropic relations.
      Returns an error message if invalid input is provided.
    """
  
    if mach < 0 or gamma <= 1:
      return "Invalid input: Mach number must be non-negative and specific heat ratio must be greater than 1."
  
    # Calculate static pressure ratio
    p_ratio = (1 + (gamma - 1) / 2 * mach**2)**(-gamma / (gamma - 1))
  
    # Calculate static temperature ratio
    t_ratio = (1 + (gamma - 1) / 2 * mach**2)**(-1)
  
    # Calculate static density ratio
    rho_ratio = (1 + (gamma - 1) / 2 * mach**2)**(-1 / (gamma - 1))
    
    # Calculate total pressure ratio
    p0_ratio = (1 + (gamma - 1) / 2 * mach**2)**(-gamma / (gamma - 1))
  
    # Calculate total temperature ratio (remains constant)
    t0_ratio = 1
  
    # Calculate the speed of sound ratio
    a_ratio = math.sqrt(t_ratio)
  
    results = {
        "Mach Number": mach,
        "Specific Heat Ratio": gamma,
        "Static Pressure Ratio (p/p0)": p_ratio,
        "Static Temperature Ratio (T/T0)": t_ratio,
        "Static Density Ratio (rho/rho0)": rho_ratio,
        "Total Pressure Ratio (p0/p0*)": p0_ratio,
        "Total Temperature Ratio (T0/T0*)": t0_ratio,
        "Speed of Sound Ratio (a/a*)": a_ratio
    }
    return results
  
  # Example usage:
  mach_number = 2.0
  gamma_value = 1.4
  
  results = isentropic_relations(mach_number, gamma_value)
  
  if isinstance(results, dict):
    for key, value in results.items():
        print(f"{key}: {value}")
  else:
      print(results) # Print the error message if input is invalid
