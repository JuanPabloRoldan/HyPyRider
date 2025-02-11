import math

class IsentropicRelationsSolver:
  
  def __init__(self, gamma=1.4):
      '''
      Initializes the isentropic shock solver with a specific heat ratio.

      Parameters
      ----------
      gamma : float
          Specific heat ratio, default is 1.4 for air.
      '''
      self.gamma = gamma

  def isentropic_relations(self, Mach):
    """
    Calculates isentropic relations for given Mach number and specific heat ratio.
  
    Parameters:
      Mach (float): Mach number.
      gamma (float): Specific heat ratio.
  
    Returns:
      A dictionary containing the isentropic relations.
      Returns an error message if invalid input is provided.
    """
    gamma = self.gamma

    if Mach < 0 or gamma <= 1:
      return "Invalid input: Mach number must be non-negative and specific heat ratio must be greater than 1."
  
    # Calculate static pressure ratio
    p_ratio = (1 + (gamma - 1) / 2 * Mach**2)**(-gamma / (gamma - 1))
  
    # Calculate static temperature ratio
    t_ratio = (1 + (gamma - 1) / 2 * Mach**2)**(-1)
  
    # Calculate static density ratio
    rho_ratio = (1 + (gamma - 1) / 2 * Mach**2)**(-1 / (gamma - 1))
  
    results = {
        "Mach Number": Mach,
        "Specific Heat Ratio": gamma,
        "Static Pressure Ratio (p/p0)": p_ratio,
        "Static Temperature Ratio (T/T0)": t_ratio,
        "Static Density Ratio (rho/rho0)": rho_ratio,
    }
    return results