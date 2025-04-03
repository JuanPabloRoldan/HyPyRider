import numpy as np

def newton_raphson_system(f, J, x0, tol=1e-6, max_iter=100, relaxation=1.0, log_file="outputs/nr_debug_log.txt"):
    """
    Newton-Raphson solver for a system of nonlinear equations with debug output.

    f: Function representing the system of equations (returns a vector).
    J: Function representing the Jacobian matrix of the system (6x6 matrix).
    x0: Initial guess for the solution (vector).
    tol: Tolerance for stopping (default is 1e-6).
    max_iter: Maximum number of iterations (default is 100).
    relaxation: Underrelaxation factor to improve convergence stability.
    log_file: Path to the log file to store debug output.

    Returns the solution vector that approximates the root.
    """
    x = np.array(x0, dtype=float)  # Initial guess

    with open(log_file, "w") as log:
        log.write("=== Starting Newton-Raphson Solver ===\n")
        log.write(f"Initial guess: {x}\n\n")

        for i in range(max_iter):
            fx = np.array(f(x))
            Jx = np.array(J(x))

            cond_number = np.linalg.cond(Jx)
            log.write(f"--- Iteration {i+1} ---\n")
            log.write(f"Current x: {x}\n")
            log.write(f"Residual f(x): {fx}\n")
            log.write(f"Condition number of J: {cond_number}\n")

            if cond_number > 1 / np.finfo(Jx.dtype).eps:
                log.write("Jacobian is singular or nearly singular. No solution found.\n\n")
                return None

            dx = np.linalg.solve(Jx, -fx)
            x_new = x + relaxation * dx

            # Clamp Mach number (M3) to reasonable range
            x_new[3] = np.clip(x_new[3], 1.0001, 100.0)

            log.write(f"Update step dx: {dx}\n")
            log.write(f"New x (after relaxation and clamping): {x_new}\n")
            log.write(f"Norm of update step: {np.linalg.norm(dx)}\n\n")

            if np.linalg.norm(dx) < tol:
                log.write(f"\nConverged to solution in {i+1} iterations:\n")
                log.write(f"Final x: {x_new}\n")
                return x_new

            x = x_new

        log.write("\nDid not converge within max iterations.\n")
        return None
    return x