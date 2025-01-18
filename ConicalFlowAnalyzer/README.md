# Conical Flow Analyzer

## Description
The Conical Flow Analyzer is a Python-based tool that numerically integrates the Taylor-Maccoll equations using the fourth-order Runge-Kutta method (RK4) for second-order ODEs. It models the conical flow field for hypersonic flow by solving for post-shock conditions using the oblique shock relations and integrating the Taylor-Maccoll equations.

This project includes capabilities to:
- Compute shock angle vs. cone angle.
- Numerically solve and plot velocity profiles.

## Key Features
- **Oblique Shock Solver**:
  - Computes post-shock conditions such as downstream Mach number and flow deflection angle.
  - Provides normalized velocity components (radial and normal).
- **Taylor-Maccoll Solver**:
  - Uses RK4 to solve second-order ODEs representing the Taylor-Maccoll system.
  - Outputs radial and normal velocity components as functions of cone angle.
- **Visualization**:
  - Generates plots of shock angle vs. cone angle.
  - Plots radial and normal velocity profiles.

---

## Getting Started

### Prerequisites
Ensure the following libraries are installed on your system:
- `numpy`
- `matplotlib`

Install them via:
```bash
pip install numpy matplotlib
```

---

### Usage
1. **Run the Script**:
   ```bash
   python main.py
   ```

2. **Interpret the Outputs**:
   - The script prints post-shock conditions (downstream Mach number, flow deflection angle, and normalized velocity components).
   - Plots of velocity profiles and shock/cone angles are displayed.

---

### Example Input Parameters
The default parameters in `main.py` are:
- **Specific Heat Ratio (`gamma`)**: 1.4 (air)
- **Freestream Mach Number (`M1`)**: 3.0
- **Shock Angle (`theta_s`)**: 30°

These parameters can be modified directly in the `main.py` script.

---

## File Descriptions

### 1. **`main.py`**
- Entry point for the Conical Flow Analyzer.
- Calls both the Oblique Shock Solver and the Taylor-Maccoll Solver.
- Generates velocity profile plots and shock/cone angle visualizations.

### 2. **`oblique_shock_solver.py`**
- Implements the oblique shock solver.
- Computes post-shock conditions such as downstream Mach number and flow deflection angle.

### 3. **`taylor_maccoll_solver.py`**
- Solves the Taylor-Maccoll equations using RK4 ODE2.
- Outputs radial and normal velocity components as functions of cone angle.

---

## Outputs
- **Shock vs. Cone Angle**: Visualization of shock wave angle versus cone angle for given parameters.

---

## How to Modify
1. **Change Input Parameters**:
   Modify the `gamma`, `M1`, and `theta_s` values in `main.py`.

2. **Update Solvers**:
   - Extend the `oblique_shock_solver.py` to include additional post-shock computations.
   - Modify `taylor_maccoll_solver.py` to handle custom stopping conditions or alternative integration methods.

---

## References
This project is part of the **HyPyRider** repository and is developed under Embry-Riddle Aeronautical University's AE595CC course.

---

## License
WIP