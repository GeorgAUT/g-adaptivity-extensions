import os
import sys
import types

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from firedrake import Mesh, Constant, sqrt, inner, Function, FunctionSpace, assemble
from firedrake.pyplot import tripcolor, triplot

from pde_solvers import get_solve_firedrake_class


# --- Options (minimal subset required by NavierStokesPDESolver.update_solver/solve) ---
opt = {
    "pde_type": "NavierStokes",
    "data_type": "randg",
    "num_gauss": 1,
    "anis_gauss": False,
    "mesh_scale": 2.2,
    "nu": 0.001,
    "timestep": 0.02,
    "num_time_steps": 10,
    "U_mean": 1.0,
    "use_mpi_optimized": False,
}

mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meshes", "cylinder_010.msh"))
mesh = Mesh(mesh_path)

Solver = get_solve_firedrake_class(opt)
solver = Solver(opt, dim=2, mesh_f=mesh)

# Workaround:
# NavierStokesPDESolver.get_pde_data currently calls get_pde_data_sample with the wrong signature
# (missing the required 'dim' argument). For now we bypass PDE-data generation and provide
# simple initial conditions directly.
solver.get_pde_data = types.MethodType(
    lambda self, _pde_params: {"u_ic": Constant((0.0, 0.0)), "p_ic": Constant(0.0)},
    solver,
)

pde_params = {}
solver.update_solver(pde_params, use_mpi_optimized=opt["use_mpi_optimized"])

u, p = solver.solve()

u_mag = sqrt(inner(u, u))

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

tripcolor(u_mag, axes=axs[0])
axs[0].set_title("Navier–Stokes: |u|")
axs[0].set_aspect("equal")

triplot(mesh, axes=axs[1])
axs[1].set_title("Mesh: cylinder_010.msh")
axs[1].set_aspect("equal")

fig.tight_layout()
plt.show()
