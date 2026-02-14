import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from firedrake import *
from firedrake.pyplot import triplot, tripcolor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import firedrake_mesh_to_PyG
from models.mesh_adaptor_model import Mesh_Adaptor

import matplotlib.tri as mtri

mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meshes", "cylinder_050.msh"))
mesh = Mesh(mesh_path)

markers = list(mesh.topology.exterior_facets.unique_markers)
need = {1, 2, 3, 4}
if not need.issubset(set(markers)):
    raise ValueError(f"Mesh boundary markers are {sorted(markers)} but this script expects {sorted(need)}")

nu_val = 0.001
dt_val = 0.02
num_steps = 250
U_mean = 1.0

nu = Constant(nu_val)
k = Constant(dt_val)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

u_trial = TrialFunction(V)
v_test = TestFunction(V)
p_trial = TrialFunction(Q)
q_test = TestFunction(Q)

u_now = Function(V)
u_next = Function(V)
u_star = Function(V)
p_now = Function(Q)
p_next = Function(Q)

n = FacetNormal(mesh)
f = Constant((0.0, 0.0))

coords = mesh.coordinates.dat.data_ro
x0 = float(coords[:, 0].min())
x1 = float(coords[:, 0].max())
y0 = float(coords[:, 1].min())
y1 = float(coords[:, 1].max())
H = y1 - y0

x, y = SpatialCoordinate(mesh)
inflow_u = as_vector((4.0 * U_mean * (y - y0) * (y1 - y) / (H * H), 0.0))

bcu = [
    DirichletBC(V, Constant((0.0, 0.0)), (1, 4)),
    DirichletBC(V, inflow_u, 2),
]
bcp = [DirichletBC(Q, Constant(0.0), 3)]


def sigma(u, p):
    return 2 * nu * sym(nabla_grad(u)) - p * Identity(len(u))


u_mid = 0.5 * (u_now + u_trial)

F1 = (
    inner((u_trial - u_now) / k, v_test) * dx
    + inner(dot(u_now, nabla_grad(u_mid)), v_test) * dx
    + inner(sigma(u_mid, p_now), sym(nabla_grad(v_test))) * dx
    + inner(p_now * n, v_test) * ds
    - inner(nu * dot(nabla_grad(u_mid), n), v_test) * ds
    - inner(f, v_test) * dx
)

a1, L1 = system(F1)

a2 = inner(nabla_grad(p_trial), nabla_grad(q_test)) * dx
L2 = inner(nabla_grad(p_now), nabla_grad(q_test)) * dx - (1.0 / k) * inner(div(u_star), q_test) * dx

a3 = inner(u_trial, v_test) * dx
L3 = inner(u_star, v_test) * dx - k * inner(nabla_grad(p_next - p_now), v_test) * dx

problem1 = LinearVariationalProblem(a1, L1, u_star, bcs=bcu)
problem2 = LinearVariationalProblem(a2, L2, p_next, bcs=bcp)
problem3 = LinearVariationalProblem(a3, L3, u_next)

solver1 = LinearVariationalSolver(problem1, solver_parameters={"ksp_type": "gmres", "pc_type": "sor"})
solver2 = LinearVariationalSolver(problem2, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"})
solver3 = LinearVariationalSolver(problem3, solver_parameters={"ksp_type": "cg", "pc_type": "sor"})

u_now.assign(Constant((0.0, 0.0)))
p_now.assign(Constant(0.0))
u_star.assign(u_now)
u_next.assign(u_now)
p_next.assign(p_now)

for step in range(num_steps):
    print("Remaining Steps: ", num_steps - step, end="\r")
    solver1.solve()
    solver2.solve()
    solver3.solve()
    u_now.assign(u_next)
    p_now.assign(p_next)

reg_coords = mesh.coordinates.copy(deepcopy=True)

wandb_run_path = "qls/250127_exp_poisson_big_square/p3qvqk8g"
model_filename = "model_best.pt"

api = wandb.Api()
run = api.run(wandb_run_path)
raw_config = {k: v for k, v in run.config.items()}
opt = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw_config.items()}

opt["wandb"] = False
opt["wandb_offline"] = True
os.environ["WANDB_MODE"] = "disabled"

if "new_model_monitor_type" not in opt:
    opt["new_model_monitor_type"] = "UM2N"
if "grand_diffusion" not in opt:
    opt["grand_diffusion"] = True
if "grand_step_size" not in opt:
    opt["grand_step_size"] = 0.1
if "grand_diffusion_steps" not in opt:
    opt["grand_diffusion_steps"] = 20

opt["new_model_monitor_type"] = "Hessian_Frob_u_tensor"

opt["device"] = torch.device("cpu")
opt["mesh_dims"] = [25, 25]

state_dict = torch.load(run.file(model_filename).download(replace=True).name, map_location="cpu")

model = Mesh_Adaptor(opt, gfe_in_c=3, lfe_in_c=3, deform_in_c=3).to(opt["device"])
model.load_state_dict(state_dict)
model.eval()

pyg_data = firedrake_mesh_to_PyG(mesh)
pyg_data.coarse_mesh = [mesh]
pyg_data.u_coarse_reg = [u_now]

Hessian_squared = 0
n_fd = FacetNormal(mesh)
V_hess = FunctionSpace(mesh, "CG", 1)
for l in range(2):
    ul = assemble(interpolate(u_now[l], V_hess))
    for i in range(2):
        for j in range(2):
            u_ij = Function(V_hess)
            v_h = TestFunction(V_hess)
            w_h = TrialFunction(V_hess)
            solve(
                w_h * v_h * dx == -outer(grad(ul), grad(v_h))[i, j] * dx + (outer(n_fd, grad(ul)) * v_h)[i, j] * ds,
                u_ij,
                bcs=[DirichletBC(V_hess, 0, "on_boundary")],
            )
            Hessian_squared += u_ij ** 2

for i in range(2):
    for j in range(2):
        p_ij = Function(V_hess)
        v_h = TestFunction(V_hess)
        w_h = TrialFunction(V_hess)
        solve(
            w_h * v_h * dx == -outer(grad(p_now), grad(v_h))[i, j] * dx + (outer(n_fd, grad(p_now)) * v_h)[i, j] * ds,
            p_ij,
            bcs=[DirichletBC(V_hess, 0, "on_boundary")],
        )
        Hessian_squared += p_ij ** 2

Hessian_frob = Function(V_hess, name="||H||_F").project(sqrt(Hessian_squared))
pyg_data.Hessian_Frob_u_tensor = torch.from_numpy(Hessian_frob.dat.data_ro.copy()).float()

with torch.no_grad():
    x_phys = model(pyg_data)

V_coords = VectorFunctionSpace(mesh, "CG", 1)
new_coordinates = Function(V_coords)
new_coordinates.dat.data[:] = x_phys.detach().cpu().numpy()
mesh.coordinates.assign(new_coordinates)

adapt_coords = mesh.coordinates.copy(deepcopy=True)

Vmag = FunctionSpace(mesh, "CG", 1)
u_mag_f = Function(Vmag)

mesh.coordinates.assign(reg_coords)
u_mag_f.project(sqrt(inner(u_now, u_now)))

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))


triang = mtri.Triangulation(
    mesh.coordinates.dat.data[:, 0],
    mesh.coordinates.dat.data[:, 1],
    mesh.coordinates.cell_node_map().values
)

tripcolor(u_mag_f, axes=axs[0, 0])
axs[0, 0].triplot(triang, linewidth=0.25, color='k')
axs[0, 0].set_title("Regular mesh: |u|")

axs[1, 0].triplot(triang, linewidth=0.25, color='k')
axs[1, 0].set_title("Regular mesh")

mesh.coordinates.assign(adapt_coords)
u_mag_f.project(sqrt(inner(u_now, u_now)))

tripcolor(u_mag_f, axes=axs[0, 1])
# triplot(mesh, linewidths=0.5, axes=axs[0, 1])


triang = mtri.Triangulation(
    mesh.coordinates.dat.data[:, 0],
    mesh.coordinates.dat.data[:, 1],
    mesh.coordinates.cell_node_map().values
)

axs[0, 1].triplot(triang, linewidth=0.25, color='k')

axs[0, 1].set_title("Adapted mesh (pretrained): |u|")

axs[1, 1].triplot(triang,  linewidth=0.25)
axs[1, 1].set_title("Adapted mesh (pretrained)")

for ax in axs.ravel():
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.tight_layout()
plt.show()

print(f"W&B run: {wandb_run_path}")
print(f"Loaded weights: {model_filename}")
print(f"Adapted {x_phys.shape[0]} nodes")
