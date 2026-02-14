import os
import sys
import torch
import wandb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    UnitCubeMesh,
    assemble,
    ds,
    dx,
    exp,
    grad,
    inner,
    interpolate,
    outer,
    pi,
    sin,
    solve,
    sqrt,
)

from models.mesh_adaptor_model import Mesh_Adaptor
from data import firedrake_mesh_to_PyG

import pyvista as pv



def _copy_overlapping_slices(dst, src):
    if dst.ndim != src.ndim:
        return dst
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[slices] = src[slices]
    return dst


def load_state_dict_zero_padded(model, checkpoint_state_dict):
    model_sd = model.state_dict()
    new_sd = {}
    copied = []
    padded = []
    skipped = []

    for k, v in model_sd.items():
        if k not in checkpoint_state_dict:
            new_sd[k] = v
            skipped.append(k)
            continue

        ckpt_v = checkpoint_state_dict[k]
        if not torch.is_tensor(ckpt_v):
            new_sd[k] = v
            skipped.append(k)
            continue

        ckpt_v = ckpt_v.to(dtype=v.dtype)
        if tuple(ckpt_v.shape) == tuple(v.shape):
            new_sd[k] = ckpt_v
            copied.append(k)
            continue

        out = torch.zeros_like(v)
        try:
            out = _copy_overlapping_slices(out, ckpt_v)
            new_sd[k] = out
            padded.append((k, tuple(ckpt_v.shape), tuple(v.shape)))
        except Exception:
            new_sd[k] = v
            skipped.append(k)

    incompatible = model.load_state_dict(new_sd, strict=False)
    return copied, padded, skipped, incompatible


def compute_hessian_frob_norm_scalar(mesh, u, dim=3):
    Hessian_squared = 0
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u_cg1 = assemble(interpolate(u, V))
    for i in range(dim):
        for j in range(dim):
            u_ij = Function(V)
            v = TestFunction(V)
            w = TrialFunction(V)
            solve(
                w * v * dx == -outer(grad(u_cg1), grad(v))[i, j] * dx + (outer(n, grad(u_cg1)) * v)[i, j] * ds,
                u_ij,
                bcs=[DirichletBC(V, 0, "on_boundary")],
            )
            Hessian_squared += u_ij ** 2

    return Function(V, name="||H||_F").project(sqrt(Hessian_squared))


def plot_pyvista_original_vs_adapted(mesh, u_reg, u_adapt, reg_coords, adapt_coords, filename):
    cell_nodes = mesh.coordinates.cell_node_map().values
    if cell_nodes.shape[1] != 4:
        raise ValueError(f"Expected tetrahedral cells with 4 nodes, got: {cell_nodes.shape}")

    cells = np.column_stack((np.full((len(cell_nodes), 1), 4), cell_nodes)).astype(np.int32)
    cell_types = np.full(len(cell_nodes), 10, dtype=np.int32)  # VTK_TETRA

    V_cg1 = FunctionSpace(mesh, "CG", 1)

    mesh.coordinates.assign(reg_coords)
    coords_reg = mesh.coordinates.dat.data_ro.copy().astype(np.float64)
    u_reg_cg1 = Function(V_cg1)
    u_reg_cg1.interpolate(u_reg)

    mesh.coordinates.assign(adapt_coords)
    coords_adapt = mesh.coordinates.dat.data_ro.copy().astype(np.float64)
    u_adapt_cg1 = Function(V_cg1)
    u_adapt_cg1.interpolate(u_adapt)

    orig_grid = pv.UnstructuredGrid(cells, cell_types, coords_reg)
    orig_grid.point_data["Solution"] = u_reg_cg1.dat.data_ro.copy()

    adapt_grid = pv.UnstructuredGrid(cells, cell_types, coords_adapt)
    adapt_grid.point_data["Solution"] = u_adapt_cg1.dat.data_ro.copy()

    plotter = pv.Plotter(shape=(2, 2), off_screen=True)

    plotter.subplot(0, 0)
    plotter.add_mesh(orig_grid, scalars="Solution", show_edges=True, opacity=0.3)
    plotter.add_text("Original Mesh")

    plotter.subplot(0, 1)
    plotter.add_mesh(adapt_grid, scalars="Solution", show_edges=True, opacity=0.3)
    plotter.add_text("Adapted Mesh")

    plotter.subplot(1, 0)
    plotter.add_points(coords_reg, render_points_as_spheres=True, point_size=4.0)
    plotter.add_text("Original points")

    plotter.subplot(1, 1)
    plotter.add_points(coords_adapt, render_points_as_spheres=True, point_size=4.0)
    plotter.add_text("Adapted points")

    plotter.link_views()
    plotter.show(screenshot=filename)
    print(f"Plot saved as {filename}")


def solve_poisson_dirichlet(mesh, f_expr, degree=1):
    V = FunctionSpace(mesh, "CG", degree)
    u = Function(V)
    v = TestFunction(V)
    w = TrialFunction(V)

    a = inner(grad(w), grad(v)) * dx
    L = f_expr * v * dx

    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    solve(a == L, u, bcs=bcs)
    return u


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
opt["pde_type"] = "Poisson"
if "data_type" not in opt:
    opt["data_type"] = "randg"
if "num_gauss" not in opt:
    opt["num_gauss"] = 2
if "anis_gauss" not in opt:
    opt["anis_gauss"] = False
if "HO_degree" not in opt:
    opt["HO_degree"] = 2
if "eval_quad_points" not in opt:
    opt["eval_quad_points"] = 21
if "use_mpi_optimized" not in opt:
    opt["use_mpi_optimized"] = False
if "new_model_monitor_type" not in opt:
    opt["new_model_monitor_type"] = "UM2N"
if "grand_diffusion" not in opt:
    opt["grand_diffusion"] = True
if "grand_step_size" not in opt:
    opt["grand_step_size"] = 0.01
if "grand_diffusion_steps" not in opt:
    opt["grand_diffusion_steps"] = 20

opt["device"] = torch.device("cpu")

opt["mesh_dims"] = [10, 10, 10]

mesh_dims = opt["mesh_dims"]
if isinstance(mesh_dims, str):
    mesh_dims = eval(mesh_dims)

if not (isinstance(mesh_dims, (list, tuple)) and len(mesh_dims) == 3):
    raise ValueError(f"Expected opt['mesh_dims'] to be [n, m, l], got: {mesh_dims}")

n = int(mesh_dims[0])
m = int(mesh_dims[1])
l = int(mesh_dims[2])

coarse_mesh = UnitCubeMesh(n - 1, m - 1, l - 1, name="coarse_mesh")
reg_coords = coarse_mesh.coordinates.copy(deepcopy=True)

x, y, z = SpatialCoordinate(coarse_mesh)
f_expr = sin(pi*x)*sin(pi*y)*sin(pi*z)#100*exp(-(x-1) ** 2 / 0.1 ** 2 - (y) ** 2 / 0.1 ** 2 - (z) ** 2 / 0.1 ** 2)
u_reg = solve_poisson_dirichlet(coarse_mesh, f_expr, degree=1)

fine_n = 12#int(opt["eval_quad_points"]) - 1
fine_mesh = UnitCubeMesh(fine_n, fine_n, fine_n, name="fine_mesh")
x_f, y_f, z_f = SpatialCoordinate(fine_mesh)


f_expr_fine = sin(pi*x_f)*sin(pi*y_f)*sin(pi*z_f)#100*exp(-(x_f - 1) ** 2 / 0.1 ** 2 - (y_f) ** 2 / 0.1 ** 2 - (z_f) ** 2 / 0.1 ** 2)
u_ref = solve_poisson_dirichlet(fine_mesh, f_expr_fine, degree=1)

V_HO_reg = FunctionSpace(coarse_mesh, "CG", int(opt["HO_degree"]))
u_reg_HO = assemble(interpolate(u_reg, V_HO_reg))
u_ref_on_reg = assemble(interpolate(u_ref, V_HO_reg))
err_reg = assemble(inner(u_reg_HO - u_ref_on_reg, u_reg_HO - u_ref_on_reg) * dx)

state_dict = torch.load(run.file(model_filename).download(replace=True).name, map_location="cpu")

dim = 3
model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt["device"])
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    copied, padded, skipped, incompatible = load_state_dict_zero_padded(model, state_dict)
    print("Loaded checkpoint with zero-padding for mismatched tensors")
    print(f"Exact-copied tensors: {len(copied)}")
    print(f"Zero-padded tensors:  {len(padded)}")
    print(f"Unchanged tensors:    {len(skipped)}")
model.eval()

pyg_data = firedrake_mesh_to_PyG(coarse_mesh)
pyg_data.coarse_mesh = [coarse_mesh]
pyg_data.u_coarse_reg = [0*u_reg]
if opt.get("new_model_monitor_type", "UM2N") == "Hessian_Frob_u_tensor":
    Hessian_frob_u = compute_hessian_frob_norm_scalar(coarse_mesh, u_reg, dim=3)
    pyg_data.Hessian_Frob_u_tensor = 0*torch.from_numpy(Hessian_frob_u.dat.data_ro.copy()).float()

with torch.no_grad():
    x_phys = model(pyg_data)

if not torch.isfinite(x_phys).all():
    raise ValueError(
        "Mesh adaptor produced non-finite coordinates. "
        "This usually means the monitor/inputs were all zeros (division by zero), "
        "or the model checkpoint is incompatible with the current input features."
    )

V_coords = coarse_mesh.coordinates.function_space()
new_coordinates = Function(V_coords)
new_coordinates.dat.data[:] = x_phys.detach().cpu().numpy()
coarse_mesh.coordinates.assign(new_coordinates)
adapt_coords = coarse_mesh.coordinates.copy(deepcopy=True)

u_adapt = solve_poisson_dirichlet(coarse_mesh, f_expr, degree=1)

V_HO_adapt = FunctionSpace(coarse_mesh, "CG", int(opt["HO_degree"]))
u_adapt_HO = assemble(interpolate(u_adapt, V_HO_adapt))
u_ref_on_adapt = assemble(interpolate(u_ref, V_HO_adapt))
err_adapt = assemble(inner(u_adapt_HO - u_ref_on_adapt, u_adapt_HO - u_ref_on_adapt) * dx)

print("--- Pretrained G-Adaptivity demo: Poisson on unit cube ---")
print(f"W&B run: {wandb_run_path}")
print(f"Mesh dims: {n} x {m} x {l}")
print(f"L2^2 error on regular mesh:  {float(err_reg):.6e}")
print(f"L2^2 error on adapted mesh:  {float(err_adapt):.6e}")

coarse_mesh.coordinates.assign(reg_coords)
coarse_mesh.coordinates.assign(adapt_coords)

plot_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "poisson_cube_mesh.png"))
plot_pyvista_original_vs_adapted(coarse_mesh, u_reg, u_adapt, reg_coords, adapt_coords, plot_filename)
