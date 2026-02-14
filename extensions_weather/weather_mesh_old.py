import xarray as xr
import gcsfs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os
import sys
import torch
from types import SimpleNamespace

from firedrake import (
    Constant,
    DirichletBC,
    FacetNormal,
    Function,
    FunctionSpace,
    RectangleMesh,
    TestFunction,
    TrialFunction,
    assemble,
    ds as ds_fd,
    dx,
    grad,
    outer,
    solve,
    sqrt,
)
from firedrake.pyplot import tripcolor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.mesh_adaptor_model import Mesh_Adaptor


def firedrake_mesh_to_pyg_minimal(mesh):
    coordinates = mesh.coordinates.dat.data_ro
    cell_node_map = mesh.coordinates.cell_node_map().values

    edges_set = set()
    for cell in cell_node_map:
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                edges_set.add((cell[i], cell[j]))
                edges_set.add((cell[j], cell[i]))

    edge_index = torch.tensor(list(edges_set), dtype=torch.long).t().contiguous()
    x_in = torch.tensor(coordinates, dtype=torch.float)

    return SimpleNamespace(x_in=x_in, edge_index=edge_index)


def compute_hessian_frob_norm_scalar(mesh, u_in, dim=2):
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    if hasattr(u_in, "function_space") and u_in.function_space() == V:
        u_cg1 = u_in
    else:
        u_cg1 = Function(V)
        u_cg1.interpolate(u_in)

    Hessian_squared = 0
    for i in range(dim):
        for j in range(dim):
            u_ij = Function(V)
            v = TestFunction(V)
            w = TrialFunction(V)
            solve(
                w * v * dx
                == -outer(grad(u_cg1), grad(v))[i, j] * dx + (outer(n, grad(u_cg1)) * v)[i, j] * ds_fd,
                u_ij,
                bcs=[DirichletBC(V, Constant(0.0), "on_boundary")],
            )
            Hessian_squared += u_ij**2

    Hessian_frob = Function(V, name="||H(u)||_F").project(sqrt(Hessian_squared))
    return Hessian_frob

fs = gcsfs.GCSFileSystem(token="anon")  # public bucket
mapper = fs.get_mapper(
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

ds_xr = xr.open_zarr(mapper, consolidated=True)
print(ds_xr)

da = ds_xr["10m_wind_speed"]
print(da)
print(da.dims)
print(da.shape)
print(da.coords)
slice0 = da.isel(time=0)
print(slice0)

t = 0
slice_t = da.isel(time=t)

downsample = 5
slice_t = slice_t.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))
print("Downsampled slice dims:", slice_t.dims)
print("Downsampled slice shape:", slice_t.shape)

lon = slice_t.longitude.values
lat = slice_t.latitude.values

nx = lon.size - 1
ny = lat.size - 1
Lx = float(lon.max() - lon.min() + (lon[1] - lon[0]))
Ly = float(lat.max() - lat.min())
mesh = RectangleMesh(nx, ny, Lx, Ly)

V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name=str(da.name))

values = np.asarray(slice_t.values)
if lat[0] > lat[-1]:
    values = values[::-1, :]

coords = mesh.coordinates.dat.data_ro
print(f"Mesh coordinate bounds: x [{coords[:, 0].min()}, {coords[:, 0].max()}], y [{coords[:, 1].min()}, {coords[:, 1].max()}]")
dx_lon = float(lon[1] - lon[0])
dy_lat = float(abs(lat[1] - lat[0]))

ii = np.rint(coords[:, 0] / dx_lon).astype(np.int64)
jj = np.rint(coords[:, 1] / dy_lat).astype(np.int64)
ii = np.clip(ii, 0, lon.size - 1)
jj = np.clip(jj, 0, lat.size - 1)

u.dat.data[:] = values[jj, ii]/5.0  # Scale down for better adaptivity
# Print min and max of u
print(f"{da.name} value bounds: [{u.dat.data_ro.min()}, {u.dat.data_ro.max()}]")

plt.figure(figsize=(14, 5))
tc = tripcolor(u, shading="gouraud", cmap="viridis")
plt.colorbar(tc, label=str(da.name))
plt.title(f"{da.name} on Firedrake CG1 mesh at time={str(slice_t['time'].values)}")
plt.xlabel("longitude (deg)")
plt.ylabel("latitude (deg)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
tc = tripcolor(u, axes=ax, shading="gouraud", cmap="viridis")

triang = mtri.Triangulation(
    mesh.coordinates.dat.data_ro[:, 0],
    mesh.coordinates.dat.data_ro[:, 1],
    mesh.coordinates.cell_node_map().values,
)
ax.triplot(triang, linewidth=0.1, color="r")

fig.colorbar(tc, ax=ax, label=str(da.name))
ax.set_title(f"{da.name} with mesh overlay at time={str(slice_t['time'].values)}")
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
fig.tight_layout()
plt.show()

opt = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_patches": False,
}

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
if 'show_mesh_evol_plots' not in opt:
    opt['show_mesh_evol_plots'] = False


opt['mesh_dims'] = [nx + 1, ny + 1]  # CG1 mesh has (nx+1) x (ny+1) vertices

model_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions", "model_best.pt"))
state_dict = torch.load(model_filename, map_location="cpu")

dim = 2
model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt["device"])
model.load_state_dict(state_dict)
model.eval()

reg_coords = mesh.coordinates.copy(deepcopy=True)

mesh.coordinates.dat.data[:, 0]*=1/360.0
mesh.coordinates.dat.data[:, 1]*=1/180.0

pyg_data = firedrake_mesh_to_pyg_minimal(mesh)
pyg_data.coarse_mesh = [mesh]
pyg_data.u_coarse_reg = [u]

# print(opt.get("new_model_monitor_type", "UM2N"))
# if opt.get("new_model_monitor_type", "UM2N") == "Hessian_Frob_u_tensor":
#     input('')
Hessian_frob_u = compute_hessian_frob_norm_scalar(mesh, u, dim=2)
pyg_data.Hessian_Frob_u_tensor = torch.relu(torch.from_numpy(Hessian_frob_u.dat.data_ro.copy()).float()/5.0-1000.0)#+1000.0

# Plot Hessian_Frob_u_tensor
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
hessian_vals = pyg_data.Hessian_Frob_u_tensor.detach().cpu().numpy()
triang_h = mtri.Triangulation(
    mesh.coordinates.dat.data_ro[:, 0],
    mesh.coordinates.dat.data_ro[:, 1],
    mesh.coordinates.cell_node_map().values,
)
hessian_tc = ax.tripcolor(triang_h, hessian_vals, shading="gouraud", cmap="inferno")
fig.colorbar(hessian_tc, ax=ax, label="||H(u)||_F")
ax.set_title("Frobenius norm of Hessian of u on CG1 mesh")
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
fig.tight_layout()
plt.show()
# Print min and max of Hessian_Frob_u_tensor
print(f"Hessian_Frob_u_tensor value bounds: [{pyg_data.Hessian_Frob_u_tensor.min().item()}, {pyg_data.Hessian_Frob_u_tensor.max().item()}]")
with torch.no_grad():
    print("Running mesh adaptor model...")
    x_phys = model(pyg_data)

if not torch.isfinite(x_phys).all():
    raise ValueError("Mesh adaptor produced non-finite coordinates")

mesh.coordinates.dat.data[:] = x_phys.detach().cpu().numpy()
adapt_coords = mesh.coordinates.copy(deepcopy=True)

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
tc = tripcolor(u, axes=ax, shading="gouraud", cmap="viridis")

triang = mtri.Triangulation(
    mesh.coordinates.dat.data_ro[:, 0],
    mesh.coordinates.dat.data_ro[:, 1],
    mesh.coordinates.cell_node_map().values,
)
ax.triplot(triang, linewidth=0.1, color="r")

fig.colorbar(tc, ax=ax, label=str(da.name))
ax.set_title(f"{da.name} on adapted mesh at time={str(slice_t['time'].values)}")
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
fig.tight_layout()
out_png = os.path.abspath(os.path.join(os.path.dirname(__file__), "weather_mesh_adapted.png"))
fig.savefig(out_png, dpi=200)
print("Saved plot:", out_png)
plt.show()

mesh.coordinates.assign(reg_coords)
mesh.coordinates.assign(adapt_coords)
