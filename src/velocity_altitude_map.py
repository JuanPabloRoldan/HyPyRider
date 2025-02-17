import numpy as np
import matplotlib.pyplot as plt

def calculate_pressure(altitude):
    '''
    Calculates atmospheric pressure based on altitude using a segmented atmospheric model.

    Parameters
    ----------
    altitude : float
        Altitude in meters.

    Returns
    -------
    float
        Atmospheric pressure in Pascals.
    '''
    if altitude <= 11000:  # Troposphere
        T = 15.04 - 0.00649 * altitude
        P = 101.29 * ((T + 273.1) / 288.08) ** 5.256
    elif 11000 < altitude <= 25000:  # Lower Stratosphere
        T = -56.46
        P = 22.65 * np.exp(1.73 - 0.000157 * altitude)
    else:  # Upper Stratosphere
        T = -131.21 + 0.00299 * altitude
        P = 2.488 * ((T + 273.1) / 216.6) ** -11.388
    return P * 1000  # Convert from kPa to Pa

def calculate_temperature(altitude):
    '''
    Calculates atmospheric temperature based on altitude.

    Parameters
    ----------
    altitude : float
        Altitude in meters.

    Returns
    -------
    float
        Temperature in Celsius.
    '''
    if altitude <= 11000:  # Troposphere
        T = 15.04 - 0.00649 * altitude
    elif 11000 < altitude <= 25000:  # Lower Stratosphere
        T = -56.46
    else:  # Upper Stratosphere
        T = -131.21 + 0.00299 * altitude
    return T

def calculate_density(pressure, temperature):
    '''
    Calculates air density using the ideal gas law.

    Parameters
    ----------
    pressure : float
        Atmospheric pressure in Pascals.
    temperature : float
        Atmospheric temperature in Celsius.

    Returns
    -------
    float
        Air density in kg/m^3.
    '''
    return pressure / (0.2869 * (temperature + 273.1))

def calculate_dynamic_pressure(gamma, p, M):
    '''
    Calculates dynamic pressure for a given Mach number, specific heat ratio, and pressure.

    Parameters
    ----------
    gamma : float
        Specific heat ratio (e.g., 1.4 for air).
    p : float
        Static pressure in Pascals.
    M : float
        Mach number.

    Returns
    -------
    float
        Dynamic pressure in Pascals.
    '''
    return 0.5 * gamma * p * M ** 2

def plot_altitude_vs_mach(gamma, dynamic_pressures):
    '''
    Plots altitude vs. Mach number for various dynamic pressures.

    Parameters
    ----------
    gamma : float
        Specific heat ratio (e.g., 1.4 for air).
    dynamic_pressures : list of float
        List of dynamic pressure values in Pascals to be plotted.

    Returns
    -------
    None
    '''
    # Create altitude range (in meters)
    altitudes = np.linspace(0, 45000, 500)

    plt.figure(figsize=(10, 8))

    for idx, q in enumerate(dynamic_pressures):
        mach_numbers = []
        for altitude in altitudes:
            pressure = calculate_pressure(altitude)
            temperature = calculate_temperature(altitude)
            density = calculate_density(pressure, temperature)
            # Calculate Mach number for given dynamic pressure (q)
            mach = np.sqrt((2 * q) / (gamma * pressure))
            mach_numbers.append(mach)

        # Convert dynamic pressure from Pascals to atm for labeling
        q_atm = q / 101325
        plt.plot(mach_numbers, altitudes / 1000, label=f"q = {q_atm:.2f} atm",
                 linewidth=2.5)

    # Add labels, legend, and grid
    plt.title("Altitude vs. Mach Number", fontsize=14)
    plt.xlabel("Mach Number", fontsize=12)
    plt.xlim(0, 20)
    plt.ylabel("Altitude (km)", fontsize=12)
    plt.ylim(0, 80)  # Enforce y-axis range from 0 to 80 km
    plt.grid(True)
    plt.legend()
    plt.show()

# # Example usage
# gamma = 1.4  # Specific heat ratio for air
# dynamic_pressures = [25331.25, 101325]  # Array of dynamic pressure values in Pascals

# plot_altitude_vs_mach(gamma, dynamic_pressures)