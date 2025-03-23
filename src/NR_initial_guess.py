# For the Newton Raphson method, we need the inital conditions. We are finding theta_3, mu_3, M3, x3, y3, and nu3. We are given
# theta_1, mu_1, M1, x1, y1, and nu1. We are also given theta_2, mu_2, M2, x2, y2, and nu2. 
# We are importing the known conditions from another file to use in this file.
import numpy as np
def find_unknown_conditions_3(theta_1, mu_1, M1, x1, y1, nu1, theta_2, mu_2, M2, x2, y2, nu2):
    # Defining the function to find the unknown conditions. All the unknown conditions are averages of the two initial conditions.

    theta_3 = (theta_1 + theta_2) / 2
    mu_3 = (mu_1 + mu_2) / 2
    M3 = (M1 + M2) / 2
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2
    nu_3 = (nu1 + nu2) / 2

    # Create xo vector for all the solved conditions.
    xo = np.array([theta_3, mu_3, M3, x3, y3, nu_3])