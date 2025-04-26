import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from moc_to_structured_grid import extract_fields_from_moc
from axi_sym_MoC_solver import MoC_Skeleton

# === MoC Setup ===
M_inf = 7.0
gamma = 1.2
T_inf = 231.64  # [K]
R = 287
a_inf = np.sqrt(gamma * R * T_inf)

wall_params = {"x1": 3.5010548, "x2": 9.39262, "r1": 3.5507, "r2": 2.5}
z1, z2 = wall_params["x1"], wall_params["x2"]
r1, r2 = wall_params["r1"], wall_params["r2"]

# === Generate MoC Mesh ===
moc_solver = MoC_Skeleton(M_inf, a_inf, gamma, wall_params)
moc_mesh = moc_solver.MoC_Mesher()
x_vals, r_vals, u_field, v_field = extract_fields_from_moc(moc_mesh)

# === Compute Mach Number ===
V_mag = np.sqrt(u_field**2 + v_field**2)
Mach_field = V_mag / a_inf

# === Flatten Valid Data ===
x_full = x_vals.flatten()
r_full = r_vals.flatten()
mach_full = Mach_field.flatten()
valid = ~np.isnan(x_full) & ~np.isnan(r_full)
x = x_full[valid]
r = r_full[valid]
mach = mach_full[valid]

# === Wall Function ===
def r_b(z):
    return ((r2 - r1) / (z2 - z1)**2) * (z - z1)**2 + r1

z_wall = np.linspace(z1, z2, 300)
r_wall = r_b(z_wall)

# === Triangulation and Geometric Masking ===
triang = tri.Triangulation(x, r)
x_tri = np.mean(x[triang.triangles], axis=1)
r_tri = np.mean(r[triang.triangles], axis=1)
r_wall_at_xtri = r_b(x_tri)
tri_mask = r_tri >= r_wall_at_xtri
triang.set_mask(~tri_mask)

# === Plotting ===
fig, ax = plt.subplots(figsize=(10, 8))

# Filled contour
tpc = ax.tripcolor(triang, mach, shading='flat', cmap='viridis')
plt.colorbar(tpc, ax=ax, label="Mach Number")

# MoC mesh points
ax.plot(x, r, 'ko', markersize=2, label="MoC nodes")

# MoC mesh lines (C+ and C-)
i_max, j_max = x_vals.shape
for i in range(i_max):
    row = [(x_vals[i, j], r_vals[i, j]) for j in range(j_max) if not np.isnan(x_vals[i, j])]
    if len(row) > 1:
        ax.plot(*zip(*row), color='black', linewidth=0.4)
for j in range(j_max):
    col = [(x_vals[i, j], r_vals[i, j]) for i in range(i_max) if not np.isnan(x_vals[i, j])]
    if len(col) > 1:
        ax.plot(*zip(*col), color='black', linewidth=0.4)

# Wall surface
ax.plot(z_wall, r_wall, color="grey", linewidth=2, label="Wall surface")

# Formatting
ax.set_xlabel('x')
ax.set_ylabel('r')
ax.set_title("Mach Number Contours with MoC Mesh and Wall")
ax.axis('equal')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
