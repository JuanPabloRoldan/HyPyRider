# HyPyRider

## Description

HyPyRider (**Hy**personic **Py**thon-based wave**Rider** design tool) is a repository for designing and analyzing a waverider through multiple sub-projects, including:

- **Conical Flow Analyzer**: Solves conical flow problems using oblique shock and Taylor-Maccoll solvers.
- **Busemann Inlet Design**: Designs optimal inlet geometry for supersonic flow.
- **Compression/Expansion Surfaces**: Analyzes flow interactions with compression and expansion surfaces.
- **Turbo Ramjet Cycle Analysis**: Simulates and evaluates turbo ramjet engine cycles.
- **Axisymmetric Method of Characteristics Analyzer**: Solves axisymmetric flow problems using the method of characteristics.
- **Hypersonic Waverider Viscous Corrections**: Incorporates viscous effects into waverider designs.
- **Hypersonic Waverider Expansion Surface Design**: Designs expansion surfaces for hypersonic flows.

Of these, the **Conical Flow Analyzer**, **Axisymmetric MoC Analyzer**, and **Compression/Expansion Surfaces** sub-projects currently have working implementations under `src/` (see [Project Structure](#project-structure) below). The Busemann Inlet Design, Turbo Ramjet Cycle Analysis, and Viscous Corrections sub-projects are planned but not yet started.

This guide provides clear instructions for setting up the project, making changes, and collaborating using Git and VS Code.

---

## Getting Started

### Prerequisites

1. **Install Git**: [Download Git](https://git-scm.com/downloads) and install it.
2. **Install Python 3.10+**: Ensure Python is installed on your system. (Tested on Python 3.12.)
3. **Install VS Code**: [Download Visual Studio Code](https://code.visualstudio.com/) and install it.
   - Install the **Python Extension** for VS Code.

---

### 1. Clone the Repository

To get started, clone the repository to your local machine:

```bash
# Clone the repository from GitHub
git clone https://github.com/JuanPabloRoldan/HyPyRider.git

# Navigate into the project directory
cd HyPyRider
```

---

### 2. Create a Branch

Before making any changes, create a new branch based on the `main` branch:

```bash
# Pull the latest changes from the main branch
git pull origin main

# Create and switch to a new branch
# Replace "your-branch-name" with a descriptive name for your branch
git checkout -b your-branch-name
```

---

### 3. Set Up a Virtual Environment and Install Dependencies

Create an isolated virtual environment so project dependencies don't conflict with anything else on your system, then install the required libraries listed in `requirements.txt`:

```bash
# Create a virtual environment (once per clone)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 4. Edit and Test the Code

1. Open the project in VS Code:

   ```bash
   code .
   ```

2. Edit the relevant module for your sub-project under `src/` (e.g. `src/conical_flow_analyzer.py`, `src/moc_solver_pc.py`).
3. Run the module's example usage directly to sanity-check your changes (most modules define one under `if __name__ == "__main__":`):

   ```bash
   python src/conical_flow_analyzer.py
   ```

---

### 5. Run the Test Suite

Automated tests live in `tests/` and run via `pytest`, configured by `pytest.ini`:

```bash
pytest
```

Any bug fix or new solver logic should keep existing tests passing and add new tests alongside it. Test coverage spans the isentropic, oblique shock, Taylor-Maccoll, and MoC point solvers, plus `point.py`, `process_LE_points.py`, `metric_derivative_solver.py`, and `velocity_altitude_map.py`. The streamline integrator and surface pressure solver still have no automated tests.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the full suite on every push and pull request against `main`, on Python 3.12 (the version `requirements.txt` is pinned against).

---

### 6. Lint Your Changes

This repo uses [ruff](https://docs.astral.sh/ruff/) for linting (unused imports/variables, import ordering, line length). Install it via the dev requirements file and run it from the repo root:

```bash
pip install -r requirements-dev.txt
ruff check .
```

`ruff check .` also runs in CI on every push and pull request.

---

### 7. Add and Commit Changes

After making and testing your changes:

1. **Stage your changes**:

   ```bash
   git add .
   ```

2. **Commit your changes**:

   ```bash
   git commit -m "Descriptive message about your changes"
   ```

---

### 8. Push Your Branch and Open a Pull Request

```bash
# Push your branch to GitHub
git push origin your-branch-name
```

Then, on GitHub: open the **Pull Requests** tab, click **New Pull Request**, select your branch against `main`, add a title/description, and submit. Your changes will be reviewed and merged into `main` by the project maintainer.

---

## Project Structure

```text
HyPyRider/
├── src/
│   ├── conical_flow_analyzer.py       # Orchestrates oblique shock + Taylor-Maccoll solvers
│   ├── oblique_shock_solver.py        # Oblique shock jump relations
│   ├── taylor_maccoll_solver.py       # RK4 integration of the Taylor-Maccoll ODE
│   ├── isentropic_relations_solver.py # Isentropic property ratios
│   ├── moc_solver_pc.py               # Predictor-corrector axisymmetric MoC point solver
│   ├── axi-sym_MoC_solver.py          # Builds a full MoC characteristic mesh
│   ├── point.py                       # Shared mesh-point value object
│   ├── streamline_integrator.py       # Traces streamlines / builds lower-surface geometry
│   ├── lower_surface_pressure_solver.py # Surface mesh Cp/Cl/Cd + VTK export
│   ├── metric_derivative_solver.py    # Grid-metric transform + characteristic-line integration
│   ├── MachCone_Vertex_Finder.py      # Mach cone vertex from leading-edge geometry
│   ├── process_LE_points.py           # Leading-edge point file parsing
│   └── velocity_altitude_map.py       # Atmospheric properties + dynamic pressure mapping
├── tests/                             # pytest test suite (see below)
├── pytest.ini                         # pytest configuration (adds src/ to the path)
├── requirements.txt                   # Pinned Python dependencies
└── README.md
```

---

## Dependencies

The required Python libraries are pinned in `requirements.txt` and installed with:

```bash
pip install -r requirements.txt
```

---

## License

TBD — no license has been chosen yet.
