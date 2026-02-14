import os
import sys
import yaml
import torch
import wandb
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from firedrake import Function, FunctionSpace, VectorFunctionSpace, UnitSquareMesh, assemble, inner, dx, interpolate
#from firedrake.__future__ import interpolate
from firedrake.pyplot import tripcolor, triplot

from models.mesh_adaptor_model import Mesh_Adaptor
from data import firedrake_mesh_to_PyG
from pde_solvers import get_solve_firedrake_class


wandb_run_path = "qls/250127_exp_poisson_big_square/p3qvqk8g"
model_filename = "model_best.pt"

api = wandb.Api()
run = api.run(wandb_run_path)

raw_config = {k: v for k, v in run.config.items()}
opt = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw_config.items()}


opt["wandb"] = False
opt["wandb_offline"] = True
os.environ["WANDB_MODE"] = "disabled"

if "mesh_geometry" not in opt:
    opt["mesh_geometry"] = "rectangle"
if "mesh_file_type" not in opt:
    opt["mesh_file_type"] = "bin"
if "pde_type" not in opt:
    opt["pde_type"] = "Poisson"
if "data_type" not in opt:
    opt["data_type"] = "randg"
if "num_gauss" not in opt:
    opt["num_gauss"] = 2
if "anis_gauss" not in opt:
    opt["anis_gauss"] = False
if "HO_degree" not in opt:
    opt["HO_degree"] = 4
if "eval_quad_points" not in opt:
    opt["eval_quad_points"] = 101
if "use_mpi_optimized" not in opt:
    opt["use_mpi_optimized"] = False
if "new_model_monitor_type" not in opt:
    opt["new_model_monitor_type"] = "UM2N"
if "grand_diffusion" not in opt:
    opt["grand_diffusion"] = True
if "grand_step_size" not in opt:
    opt["grand_step_size"] = 0.1
if "grand_diffusion_steps" not in opt:
    opt["grand_diffusion_steps"] = 20

if torch.backends.mps.is_available():
    opt["device"] = torch.device('cpu') #mps
elif torch.cuda.is_available():
    opt["device"] = torch.device('cpu')
else:
    opt["device"] = torch.device("cpu")

if "mesh_dims" not in opt:
    opt["mesh_dims"] = [25, 25]

mesh_dims = opt["mesh_dims"]
if isinstance(mesh_dims, str):
    mesh_dims = eval(mesh_dims)

if isinstance(mesh_dims, (list, tuple)) and len(mesh_dims) == 2:
    n = int(mesh_dims[0])
    m = int(mesh_dims[1])
else:
    raise ValueError(f"Expected opt['mesh_dims'] to be [n, m], got: {mesh_dims}")

coarse_mesh = UnitSquareMesh(n - 1, m - 1, name="coarse_mesh")
reg_coords = coarse_mesh.coordinates.copy(deepcopy=True)

Solver = get_solve_firedrake_class(opt)
coarse_solver = Solver(opt, 2, coarse_mesh)

pde_params = coarse_solver.get_pde_params(idx=0, num_data=1, num_gaus=int(opt["num_gauss"]))
if "mon_power" in opt:
    pde_params["mon_power"] = opt["mon_power"]
if "monitor_type" in opt:
    pde_params["monitor_type"] = opt["monitor_type"]
if "mon_reg" in opt:
    pde_params["mon_reg"] = opt["mon_reg"]
pde_params["num_gauss"] = int(opt["num_gauss"])

coarse_solver.update_solver(pde_params)
u_reg = coarse_solver.solve(use_mpi_optimized=opt["use_mpi_optimized"])

# Mesh_Adaptor.process_batch requires either:
# - UM2N: data.u_coarse_reg[0] (a Firedrake Function)
# - Hessian: data.Hessian_Frob_u_tensor (a torch tensor)
# and always uses: data.coarse_mesh[0]
if opt.get("new_model_monitor_type", "UM2N") == "Hessian_Frob_u_tensor":
    Hessian_Frob_u = coarse_solver.get_Hessian_Frob_norm()
    Hessian_Frob_u_tensor = torch.from_numpy(Hessian_Frob_u.dat.data).float()

fine_mesh = UnitSquareMesh(int(opt["eval_quad_points"]) - 1, int(opt["eval_quad_points"]) - 1, name="fine_mesh")
fine_solver = Solver(opt, 2, fine_mesh)
fine_solver.update_solver(pde_params)
u_ref = fine_solver.solve(use_mpi_optimized=opt["use_mpi_optimized"])

V_HO_reg = FunctionSpace(coarse_mesh, "CG", int(opt["HO_degree"]))
u_reg_HO = assemble(interpolate(u_reg, V_HO_reg))
u_ref_on_reg = assemble(interpolate(u_ref, V_HO_reg))
err_reg = assemble(inner(u_reg_HO - u_ref_on_reg, u_reg_HO - u_ref_on_reg) * dx)

state_dict = torch.load(run.file(model_filename).download(replace=True).name, map_location="cpu")

dim = 2
model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt["device"])
model.load_state_dict(state_dict)
model.eval()

pyg_data = firedrake_mesh_to_PyG(coarse_mesh)
pyg_data.coarse_mesh = [coarse_mesh]
pyg_data.u_coarse_reg = [u_reg]
if opt.get("new_model_monitor_type", "UM2N") == "Hessian_Frob_u_tensor":
    pyg_data.Hessian_Frob_u_tensor = Hessian_Frob_u_tensor

with torch.no_grad():
    x_phys = model(pyg_data)

V_coords = VectorFunctionSpace(coarse_mesh, "CG", 1)
new_coordinates = Function(V_coords)
new_coordinates.dat.data[:] = x_phys.detach().cpu().numpy()
coarse_mesh.coordinates.assign(new_coordinates)
adapt_coords = coarse_mesh.coordinates.copy(deepcopy=True)

coarse_solver.update_solver(pde_params)
u_adapt = coarse_solver.solve(use_mpi_optimized=opt["use_mpi_optimized"])

V_HO_adapt = FunctionSpace(coarse_mesh, "CG", int(opt["HO_degree"]))
u_adapt_HO = assemble(interpolate(u_adapt, V_HO_adapt))
u_ref_on_adapt = assemble(interpolate(u_ref, V_HO_adapt))
err_adapt = assemble(inner(u_adapt_HO - u_ref_on_adapt, u_adapt_HO - u_ref_on_adapt) * dx)

print("--- Pretrained G-Adaptivity demo: Poisson on unit square ---")
print(f"W&B run: {wandb_run_path}")
print(f"Mesh dims: {n} x {m}")
print(f"L2^2 error on regular mesh:  {float(err_reg):.6e}")
print(f"L2^2 error on adapted mesh:  {float(err_adapt):.6e}")
if float(err_reg) > 0:
    print(f"Relative change (%): {(float(err_adapt) - float(err_reg)) / float(err_reg) * 100.0:.2f}")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

coarse_mesh.coordinates.assign(reg_coords)
tripcolor(u_reg, axes=axs[0, 0])
_artists = triplot(coarse_mesh, axes=axs[0, 0])
# try:
#     for _a in _artists:
#         _a.set_color("k")
#         _a.set_linewidth(0.4)
# except TypeError:
#     pass
axs[0, 0].set_title("Regular mesh: solution")

coarse_mesh.coordinates.assign(adapt_coords)
tripcolor(u_adapt, axes=axs[0, 1])
_artists = triplot(coarse_mesh, axes=axs[0, 1])
# try:
#     for _a in _artists:
#         _a.set_color("k")
#         _a.set_linewidth(0.4)
# except TypeError:
#     pass
axs[0, 1].set_title("Adapted mesh: solution")

coarse_mesh.coordinates.assign(reg_coords)
_artists = triplot(coarse_mesh, axes=axs[1, 0])
# try:
#     for _a in _artists:
#         _a.set_color("k")
#         _a.set_linewidth(0.4)
# except TypeError:
#     pass
axs[1, 0].set_title("Regular mesh")

coarse_mesh.coordinates.assign(adapt_coords)
_artists = triplot(coarse_mesh, axes=axs[1, 1])
# try:
#     for _a in _artists:
#         _a.set_color("k")
#         _a.set_linewidth(0.4)
# except TypeError:
#     pass
axs[1, 1].set_title("Adapted mesh")

for ax in axs.ravel():
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.tight_layout()
plt.show()
