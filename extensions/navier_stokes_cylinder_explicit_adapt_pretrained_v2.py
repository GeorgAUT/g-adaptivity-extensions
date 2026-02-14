import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import imageio.v2 as imageio

from firedrake import *
from firedrake.pyplot import triplot, tripcolor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import firedrake_mesh_to_PyG
from models.mesh_adaptor_model import Mesh_Adaptor

import matplotlib.tri as mtri


def compute_hessian_frob_norm(mesh, u_now, p_now, dim=2):
    Hessian_squared = 0
    n_fd = FacetNormal(mesh)
    V_hess = FunctionSpace(mesh, "CG", 1)

    for l in range(dim):
        ul = assemble(interpolate(u_now[l], V_hess))
        for i in range(dim):
            for j in range(dim):
                u_ij = Function(V_hess)
                v_h = TestFunction(V_hess)
                w_h = TrialFunction(V_hess)
                solve(
                    w_h * v_h * dx
                    == -outer(grad(ul), grad(v_h))[i, j] * dx + (outer(n_fd, grad(ul)) * v_h)[i, j] * ds,
                    u_ij,
                    bcs=[DirichletBC(V_hess, 0, "on_boundary")],
                )
                Hessian_squared += u_ij ** 2

    for i in range(dim):
        for j in range(dim):
            p_ij = Function(V_hess)
            v_h = TestFunction(V_hess)
            w_h = TrialFunction(V_hess)
            solve(
                w_h * v_h * dx
                == -outer(grad(p_now), grad(v_h))[i, j] * dx + (outer(n_fd, grad(p_now)) * v_h)[i, j] * ds,
                p_ij,
                bcs=[DirichletBC(V_hess, 0, "on_boundary")],
            )
            Hessian_squared += p_ij ** 2

    return Function(V_hess, name="||H||_F").project(sqrt(Hessian_squared))

mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meshes", "cylinder_050.msh"))
mesh = Mesh(mesh_path)
# Print min and max coordinates
coords = mesh.coordinates.dat.data_ro
print(f"Mesh coordinate bounds: x [{coords[:, 0].min()}, {coords[:, 0].max()}], y [{coords[:, 1].min()}, {coords[:, 1].max()}]")
ghost_mesh = Mesh(mesh_path)

markers = list(mesh.topology.exterior_facets.unique_markers)
need = {1, 2, 3, 4}
if not need.issubset(set(markers)):
    raise ValueError(f"Mesh boundary markers are {sorted(markers)} but this script expects {sorted(need)}")

nu_val = 0.001
dt_val = 0.02
num_steps = 250
U_mean = 1.0

frame_every = 1

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

ghost_reg_coords = ghost_mesh.coordinates.copy(deepcopy=True)
pyg_data_ghost = firedrake_mesh_to_PyG(ghost_mesh)
pyg_data_ghost.coarse_mesh = [ghost_mesh]

V_ghost = VectorFunctionSpace(ghost_mesh, "CG", 2)
Vmag_ghost = FunctionSpace(ghost_mesh, "CG", 1)
u_ghost = Function(V_ghost)
u_mag_ghost = Function(Vmag_ghost)

video_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_adapted_u.gif")
)
fps = 10
fig_vid, ax_vid = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer = imageio.get_writer(video_path, mode="I", duration=1.0 / fps)

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

    if step % frame_every == 0:

        ghost_mesh.coordinates.assign(ghost_reg_coords)

        Hessian_frob = compute_hessian_frob_norm(mesh, u_now, p_now, dim=2)
        pyg_data_ghost.Hessian_Frob_u_tensor = torch.from_numpy(Hessian_frob.dat.data_ro.copy()).float()
        # Print min and max of Hessian_frob
        print(f"Step {step}: ||H||_F bounds: [{Hessian_frob.dat.data.min()}, {Hessian_frob.dat.data.max()}]")


        coords_np = ghost_mesh.coordinates.dat.data_ro.copy()
        coords_t = torch.from_numpy(coords_np).float()
        pyg_data_ghost.x_in = coords_t
        pyg_data_ghost.x_comp = coords_t

        with torch.no_grad():
            x_phys = model(pyg_data_ghost)

        V_coords = VectorFunctionSpace(ghost_mesh, "CG", 1)
        new_coordinates = Function(V_coords)
        new_coordinates.dat.data[:] = x_phys.detach().cpu().numpy()
        ghost_mesh.coordinates.assign(new_coordinates)

        u_ghost.dat.data[:] = u_now.dat.data_ro
        u_mag_ghost.project(sqrt(inner(u_ghost, u_ghost)))

        Vmag = FunctionSpace(mesh, "CG", 1)
        u_mag_f = Function(Vmag)
        u_mag_f.project(sqrt(inner(u_now, u_now)))

        # Print min and max of u_mag_f
        print(f"Step {step}: |u| bounds: [{u_mag_f.dat.data.min()}, {u_mag_f.dat.data.max()}]")

        ax_vid.clear()
        tripcolor(u_mag_f, axes=ax_vid)
        triang = mtri.Triangulation(
            ghost_mesh.coordinates.dat.data[:, 0],
            ghost_mesh.coordinates.dat.data[:, 1],
            ghost_mesh.coordinates.cell_node_map().values,
        )
        ax_vid.triplot(triang, linewidth=0.25, color="k")
        ax_vid.set_aspect("equal")
        ax_vid.set_title(f"Adapted mesh (pretrained): |u|  t={(step + 1) * float(dt_val):.3f}")
        fig_vid.tight_layout()
        fig_vid.canvas.draw()
        rgba = np.asarray(fig_vid.canvas.buffer_rgba())
        img = rgba[:, :, :3].copy()
        writer.append_data(img)

reg_coords = mesh.coordinates.copy(deepcopy=True)

pyg_data = firedrake_mesh_to_PyG(mesh)
pyg_data.coarse_mesh = [mesh]
pyg_data.u_coarse_reg = [u_now]
Hessian_frob = compute_hessian_frob_norm(mesh, u_now, p_now, dim=2)
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

writer.close()
plt.close(fig_vid)

print(f"Wrote video to: {video_path}")

print(f"W&B run: {wandb_run_path}")
print(f"Loaded weights: {model_filename}")
print(f"Adapted {x_phys.shape[0]} nodes")
