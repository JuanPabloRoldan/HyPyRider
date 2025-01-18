# HyPyRider

## Description
HyPyRider is a Python-based repository for designing a waverider through multiple sub-projects, including:
- **Conical Flow Analyzer**: Solves conical flow problems using oblique shock and Taylor-Maccoll solvers.
- **Busemann Inlet Design**: Designs optimal inlet geometry for supersonic flow.
- **Compression/Expansion Surfaces**: Analyzes flow interactions with compression and expansion surfaces.
- **Turbo Ramjet Cycle Analysis**: Simulates and evaluates turbo ramjet engine cycles.
- **Axisymmetric Method of Characteristics Analyzer**: Solves axisymmetric flow problems using the method of characteristics.
- **Hypersonic Waverider Viscous Corrections**: Incorporates viscous effects into waverider designs.
- **Hypersonic Waverider Expansion Surface Design**: Designs expansion surfaces for hypersonic flows.

This guide provides clear instructions for setting up the project, making changes, and collaborating using Git and VS Code.

---

## Getting Started

### Prerequisites
1. **Install Git**: [Download Git](https://git-scm.com/downloads) and install it.
2. **Install Python 3.8+**: Ensure Python is installed on your system.
3. **Install VS Code**: [Download Visual Studio Code](https://code.visualstudio.com/) and install it.
   - Install the **Python Extension** for VS Code.
4. **Install Required Python Libraries**: The required libraries are listed in `requirements.txt`.

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

### 3. Install Dependencies
Install the required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 4. Edit and Test the Code
1. Open the project in VS Code:
   ```bash
   code .
   ```
2. Edit the relevant script for your sub-project (e.g., `conical_flow_analyzer/main.py` or `busemann_inlet_design.py`).
3. Test your changes by running the appropriate script:
   ```bash
   python conical_flow_analyzer/main.py
   ```

---

### 5. Add and Commit Changes
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

### 6. Push Your Branch
Push your branch to the remote repository:

```bash
# Push your branch to GitHub
git push origin your-branch-name
```

---

### 7. Create a Pull Request
1. Go to the GitHub repository in your browser.
2. Navigate to the **Pull Requests** tab.
3. Click **New Pull Request**.
4. Select your branch and compare it to `main`.
5. Add a title and description for your pull request.
6. Submit the pull request.

Your changes will now be reviewed and merged into the main branch by the project maintainer.

---

## Project Structure
```
HyPyRider/
│
├── conical_flow_analyzer/       # Sub-project: Conical Flow Analyzer
│   ├── main.py                  # Main script for the analyzer
│   ├── oblique_shock_solver.py  # Oblique shock solver code
│   ├── taylor_maccoll_solver.py # Taylor-Maccoll solver code
├── busemann_inlet_design/       # Sub-project: Busemann Inlet Design
├── compression_surfaces/        # Sub-project: Compression/Expansion Surfaces
├── turbo_ramjet_cycle/          # Sub-project: Turbo Ramjet Cycle Analysis
├── axisymmetric_characteristics/  # Sub-project: Axisymmetric Method of Characteristics Analyzer
├── hypersonic_viscous_corrections/   # Sub-project: Hypersonic Waverider Viscous Corrections
├── hypersonic_expansion_surface/     # Sub-project: Hypersonic Expansion Surface Design
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## Collaboration Workflow
1. **Clone the repo**: `git clone https://github.com/JuanPabloRoldan/HyPyRider.git`
2. **Create a branch**: `git checkout -b your-branch-name`
3. **Make changes and test locally**.
4. **Commit and push your changes**: `git commit -m "message"` and `git push origin your-branch-name`.
5. **Submit a pull request** on GitHub.

---

## Dependencies
The required Python libraries are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

---

## License
WIP
