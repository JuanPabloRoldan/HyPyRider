import numpy as np

def newton_raphson_system(f, J, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson solver for a system of nonlinear equations.

    f: Function representing the system of equations (returns a vector).
    J: Function representing the Jacobian matrix of the system (6x6 matrix).
    x0: Initial guess for the solution (12x1 vector).
    tol: Tolerance for stopping (default is 1e-6).
    max_iter: Maximum number of iterations (default is 100).
    
    Returns the solution vector that approximates the root.
    """
    x = np.array(x0, dtype=float)  # Initial guess as numpy array
    
    for i in range(max_iter):
        fx = np.array(f(x))  # Evaluate the function at current guess
        Jx = np.array(J(x))  # Evaluate the Jacobian at current guess
        
        # Check if the Jacobian is singular (determinant = 0)
        if np.linalg.cond(Jx) > 1 / np.finfo(Jx.dtype).eps:
            print("Jacobian is singular. No solution found.")
            return None
        
        # Compute the update step using the inverse of the Jacobian
        dx = np.linalg.solve(Jx, -fx)
        
        # Update the solution vector
        x_new = x + dx
        
        # Check for convergence (based on the norm of the update)
        if np.linalg.norm(dx) < tol:
            print(f"Converged to {x_new} after {i+1} iterations.")
            return x_new
        
        x = x_new  # Update guess
    
    print(f"Did not converge within {max_iter} iterations.")
    return x
