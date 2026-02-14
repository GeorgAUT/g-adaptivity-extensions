import xarray as xr
import gcsfs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os
import sys
import torch
import imageio.v2 as imageio
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

# North Atlantic region bounds
# na_lat_min = 20.0
# na_lat_max = 70.0
# na_lon_min = -85.0
# na_lon_max = 20.0

# # UK region bounds
# na_lat_max = 61.0
# na_lat_min = 49.0
# na_lon_min = -8.0
# na_lon_max = 2.0

na_lat_min = -90.0
na_lat_max =  90.0
na_lon_min = -180.0
na_lon_max =  180.0   # (or 0.0 to 360.0, depending on your convention)

downsample = 8

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
    opt["new_model_monitor_type"] = "Hessian_Frob_u_tensor"
if "grand_diffusion" not in opt:
    opt["grand_diffusion"] = True
if "grand_step_size" not in opt:
    opt["grand_step_size"] = 0.01
if "grand_diffusion_steps" not in opt:
    opt["grand_diffusion_steps"] = 20
if "show_mesh_evol_plots" not in opt:
    opt["show_mesh_evol_plots"] = False

model_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions", "model_best.pt"))
state_dict = torch.load(model_filename, map_location="cpu")


def get_north_atlantic_slice(data_array, t_idx):
    slice_t = data_array.isel(time=t_idx)
    lon = slice_t.longitude
    use_0_360 = bool(lon.min() >= 0.0 and lon.max() > 180.0)
    slice_t_na = slice_t
    if use_0_360:
        slice_t_na = slice_t_na.assign_coords(longitude=(((slice_t_na.longitude + 180.0) % 360.0) - 180.0))
    slice_t_na = slice_t_na.sortby("longitude")
    slice_t_na = slice_t_na.sortby("latitude")
    slice_t_na = slice_t_na.sel(latitude=slice(na_lat_min, na_lat_max), longitude=slice(na_lon_min, na_lon_max))
    slice_t_na = slice_t_na.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))
    return slice_t_na


slice_ref = get_north_atlantic_slice(da, 85710)
lon = slice_ref.longitude.values
lat = slice_ref.latitude.values
nx = lon.size - 1
ny = lat.size - 1
dx_lon = float(lon[1] - lon[0])
dy_lat = float(lat[1] - lat[0])
Lx = float(lon.max() - lon.min())
Ly = float(lat.max() - lat.min())

mesh = RectangleMesh(nx, ny, Lx, Ly)
mesh.coordinates.dat.data[:, 0] += float(lon.min())
mesh.coordinates.dat.data[:, 1] += float(lat.min())
reg_coords = mesh.coordinates.copy(deepcopy=True)

# opt["mesh_dims"] = [nx + 1, ny + 1]
opt["mesh_dims"] = [25, 25]
# -----Manually set opt

opt['lr'] = 0.001
opt['nu'] = 0.001
opt['dec'] = 'identity'
opt['enc'] = 'identity'
opt['seed'] = 1173
opt['decay'] = 0
opt['model'] = 'UM2N_T'
opt['scale'] = 0.2
opt['wandb'] = False
opt['center'] = 0.5
opt['device'] = torch.device(type='cpu')
opt['epochs'] = 300
opt['evaler'] = 'analytical'
opt['dataset'] = 'fd_ma_2d'
opt['dropout'] = 0
opt['loss_fn'] = 'mse'
opt['mon_reg'] = 0.1
opt['non_lin'] = 'identity'
opt['M2N_beta'] = None
opt['boundary'] = 'dirichlet'
opt['data_dir'] = '../data'
opt['num_test'] = 125
opt['pde_type'] = 'Poisson'
opt['reg_skew'] = False
opt['residual'] = True
opt['timestep'] = 0.02
opt['HO_degree'] = 4
opt['M2N_alpha'] = None
opt['conv_type'] = 'GRAND_plus'
opt['data_name'] = 'Poisson_randg_mix_2d_train_275_UM2N_MA_approx_slow_rectangle_15_0.1reg_2_3gauss_iso_bin'
opt['data_type'] = 'randg_mix'
opt['gnn_drift'] = None
opt['loss_type'] = 'pde_loss_regularised'
opt['mesh_dims'] = [25, 25]
opt['mesh_type'] = 'UM2N_MA_approx_slow'
opt['mon_power'] = 0.2
opt['num_gauss'] = 2
opt['num_train'] = 275
opt['test_frac'] = None
opt['time_step'] = 0.1
opt['UM2N_alpha'] = 5
opt['anis_gauss'] = False
opt['batch_size'] = 1
opt['hidden_dim'] = 8
opt['learn_step'] = False
opt['num_layers'] = 4
opt['rand_gauss'] = False
opt['self_loops'] = False
opt['share_conv'] = True
opt['show_plots'] = True
opt['train_frac'] = None
opt['gnn_QK_init'] = 'none'
opt['mesh_params'] = 'internal'
opt['overfit_num'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
opt['wandb_group'] = 'testing'
opt['wandb_sweep'] = True
opt['fix_boundary'] = True
opt['gnn_hyper_QK'] = False
opt['softmax_temp'] = 2
opt['wandb_entity'] = 'qls'
opt['GNN_diffusion'] = True
opt['gat_plus_type'] = 'GAT_res_lap'
opt['gnn_normalize'] = False
opt['mesh_geometry'] = 'rectangle'
opt['start_from_ma'] = False
opt['wandb_exp_idx'] = 4
opt['wandb_offline'] = True
opt['wandb_project'] = 'MovingMeshGNN'
opt['burgers_limits'] = 3
opt['gnn_dont_train'] = False
opt['gnn_inc_feat_f'] = True
opt['mesh_dims_test'] = [[12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]]
opt['mesh_file_type'] = 'bin'
opt['num_time_steps'] = 10
opt['um2n_laplacian'] = False
opt['wandb_run_name'] = None
opt['data_train_test'] = 'train'
opt['diffusion_steps'] = 32
opt['gauss_amplitude'] = 0.25
opt['global_feat_dim'] = 8
opt['gnn_inc_feat_uu'] = False
opt['grand_step_size'] = 0.1
opt['mesh_dims_train'] = [[15, 15], [20, 20]]
opt['num_gauss_range'] = [2, 3]
opt['show_mesh_plots'] = False
opt['wandb_log_plots'] = True
opt['wandb_pivot_dir'] = '../wandb_analysis'
opt['eval_quad_points'] = 101
opt['fast_M2N_monitor'] = 'slow'
opt['gaus_drift_scale'] = 5
opt['load_quad_points'] = 101
opt['wandb_save_model'] = True
opt['softmax_temp_type'] = None
opt['stiff_quad_points'] = 3
opt['burgers_rescale_ic'] = 0.6
opt['num_transformer_in'] = 3
opt['pretrained_weights'] = None
opt['show_dataset_plots'] = False
opt['gnn_inc_glob_feat_f'] = True
opt['num_transformer_out'] = 16
opt['plots_mesh_movement'] = False
opt['wandb_artifact_path'] = None
opt['gnn_inc_glob_feat_uu'] = True
opt['plots_multistep_eval'] = False
opt['show_mesh_evol_plots'] = True
opt['eval_refinement_level'] = 2
opt['num_transformer_heads'] = 4
opt['show_train_evol_plots'] = True
opt['wandb_checkpoint_freq'] = None
opt['new_model_monitor_type'] = 'Hessian_Frob_u_tensor'
opt['num_transformer_layers'] = 1
opt['loss_regulariser_weight'] = 1
opt['num_transformer_embed_dim'] = 64
opt['transformer_training_mask'] = False
opt['transformer_attention_training_mask'] = False
opt['transformer_key_padding_training_mask'] = False
opt['transformer_training_mask_ratio_lower_bound'] = 0.5
opt['transformer_training_mask_ratio_upper_bound'] = 0.9
opt['grand_diffusion'] = True
opt['grand_diffusion_steps'] = 20
opt['lr'] = 0.001
opt['nu'] = 0.001
opt['dec'] = 'identity'
opt['enc'] = 'identity'
opt['seed'] = 1173
opt['decay'] = 0
opt['model'] = 'UM2N_T'
opt['scale'] = 0.2
opt['wandb'] = False
opt['center'] = 0.5
opt['device'] = torch.device(type='cpu')
opt['epochs'] = 300
opt['evaler'] = 'analytical'
opt['dataset'] = 'fd_ma_2d'
opt['dropout'] = 0
opt['loss_fn'] = 'mse'
opt['mon_reg'] = 0.1
opt['non_lin'] = 'identity'
opt['M2N_beta'] = None
opt['boundary'] = 'dirichlet'
opt['data_dir'] = '../data'
opt['num_test'] = 125
opt['pde_type'] = 'Poisson'
opt['reg_skew'] = False
opt['residual'] = True
opt['timestep'] = 0.02
opt['HO_degree'] = 4
opt['M2N_alpha'] = None
opt['conv_type'] = 'GRAND_plus'
opt['data_name'] = 'Poisson_randg_mix_2d_train_275_UM2N_MA_approx_slow_rectangle_15_0.1reg_2_3gauss_iso_bin'
opt['data_type'] = 'randg_mix'
opt['gnn_drift'] = None
opt['loss_type'] = 'pde_loss_regularised'
opt['mesh_dims'] = [25, 25]
opt['mesh_type'] = 'UM2N_MA_approx_slow'
opt['mon_power'] = 0.2
opt['num_gauss'] = 2
opt['num_train'] = 275
opt['test_frac'] = None
opt['time_step'] = 0.1
opt['UM2N_alpha'] = 5
opt['anis_gauss'] = False
opt['batch_size'] = 1
opt['hidden_dim'] = 8
opt['learn_step'] = False
opt['num_layers'] = 4
opt['rand_gauss'] = False
opt['self_loops'] = False
opt['share_conv'] = True
opt['show_plots'] = True
opt['train_frac'] = None
opt['gnn_QK_init'] = 'none'
opt['mesh_params'] = 'internal'
opt['overfit_num'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
opt['wandb_group'] = 'testing'
opt['wandb_sweep'] = True
opt['fix_boundary'] = True
opt['gnn_hyper_QK'] = False
opt['softmax_temp'] = 2
opt['wandb_entity'] = 'qls'
opt['GNN_diffusion'] = True
opt['gat_plus_type'] = 'GAT_res_lap'
opt['gnn_normalize'] = False
opt['mesh_geometry'] = 'rectangle'
opt['start_from_ma'] = False
opt['wandb_exp_idx'] = 4
opt['wandb_offline'] = True
opt['wandb_project'] = 'MovingMeshGNN'
opt['burgers_limits'] = 3
opt['gnn_dont_train'] = False
opt['gnn_inc_feat_f'] = True
opt['mesh_dims_test'] = [[12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]]
opt['mesh_file_type'] = 'bin'
opt['num_time_steps'] = 10
opt['um2n_laplacian'] = False
opt['wandb_run_name'] = None
opt['data_train_test'] = 'train'
opt['diffusion_steps'] = 32
opt['gauss_amplitude'] = 0.25
opt['global_feat_dim'] = 8
opt['gnn_inc_feat_uu'] = False
opt['grand_step_size'] = 0.1
opt['mesh_dims_train'] = [[15, 15], [20, 20]]
opt['num_gauss_range'] = [2, 3]
opt['show_mesh_plots'] = False
opt['wandb_log_plots'] = True
opt['wandb_pivot_dir'] = '../wandb_analysis'
opt['eval_quad_points'] = 101
opt['fast_M2N_monitor'] = 'slow'
opt['gaus_drift_scale'] = 5
opt['load_quad_points'] = 101
opt['wandb_save_model'] = True
opt['softmax_temp_type'] = None
opt['stiff_quad_points'] = 3
opt['burgers_rescale_ic'] = 0.6
opt['num_transformer_in'] = 3
opt['pretrained_weights'] = None
opt['show_dataset_plots'] = False
opt['gnn_inc_glob_feat_f'] = True
opt['num_transformer_out'] = 16
opt['plots_mesh_movement'] = False
opt['wandb_artifact_path'] = None
opt['gnn_inc_glob_feat_uu'] = True
opt['plots_multistep_eval'] = False
opt['show_mesh_evol_plots'] = True
opt['eval_refinement_level'] = 2
opt['num_transformer_heads'] = 4
opt['show_train_evol_plots'] = True
opt['wandb_checkpoint_freq'] = None
opt['new_model_monitor_type'] = 'Hessian_Frob_u_tensor'
opt['num_transformer_layers'] = 1
opt['loss_regulariser_weight'] = 1
opt['num_transformer_embed_dim'] = 64
opt['transformer_training_mask'] = False
opt['transformer_attention_training_mask'] = False
opt['transformer_key_padding_training_mask'] = False
opt['transformer_training_mask_ratio_lower_bound'] = 0.5
opt['transformer_training_mask_ratio_upper_bound'] = 0.9
opt['grand_diffusion'] = True
opt['grand_diffusion_steps'] = 20



# ------
dim = 2
model = Mesh_Adaptor(opt, gfe_in_c=dim + 1, lfe_in_c=dim + 1, deform_in_c=dim + 1).to(opt["device"])
model.load_state_dict(state_dict)
model.eval()


V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name=str(da.name))

pyg_template = firedrake_mesh_to_pyg_minimal(mesh)
pyg_template.coarse_mesh = [mesh]

frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "frames_weather_mesh_adapted"))
os.makedirs(frames_dir, exist_ok=True)

atlas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "atlas2.png"))
atlas_img = plt.imread(atlas_path)

frame_paths = []
for t_idx in range(85680, 85795): #range(85680, 85720):
    print(f"Processing time index: {t_idx}")
    slice_t_na = get_north_atlantic_slice(da, t_idx)
    values = np.asarray(slice_t_na.values)

    coords = mesh.coordinates.dat.data_ro
    ii = np.rint((coords[:, 0] - float(lon.min())) / dx_lon).astype(np.int64)
    jj = np.rint((coords[:, 1] - float(lat.min())) / dy_lat).astype(np.int64)
    ii = np.clip(ii, 0, lon.size - 1)
    jj = np.clip(jj, 0, lat.size - 1)
    u.dat.data[:] = values[jj, ii] / 10.0 - 0.2#np.maximum(1.0, values[jj, ii] / 10.0 - 0.2)

    # Print min and max of u
    print(f"{da.name} value bounds at t={t_idx}: [{u.dat.data_ro.min()}, {u.dat.data_ro.max()}]")

    pyg_data = pyg_template
    pyg_data.u_coarse_reg = [u]

    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords_norm[:, 0] - float(lon.min())) / float(lon.max() - lon.min())
    coords_norm[:, 1] = (coords_norm[:, 1] - float(lat.min())) / float(lat.max() - lat.min())
    pyg_data.x_in = torch.from_numpy(coords_norm).float()

    if opt.get("new_model_monitor_type", "UM2N") == "Hessian_Frob_u_tensor":
        Hessian_frob_u = compute_hessian_frob_norm_scalar(mesh, u, dim=2)
        pyg_data.Hessian_Frob_u_tensor = torch.from_numpy(Hessian_frob_u.dat.data_ro.copy()).float()*10000.0
        # Print min and max of Hessian_Frob_u_tensor
        print(f"Hessian_Frob_u_tensor value bounds at t={t_idx}: [{pyg_data.Hessian_Frob_u_tensor.min().item()}, {pyg_data.Hessian_Frob_u_tensor.max().item()}]")
    with torch.no_grad():
        x_phys = model(pyg_data)
    if not torch.isfinite(x_phys).all():
        raise ValueError(f"Non-finite predicted coordinates at t={t_idx}")

    mesh.coordinates.dat.data[:] = x_phys.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(
        atlas_img,
        extent=[na_lon_min, na_lon_max, na_lat_min, na_lat_max],
        origin="upper",
        aspect="auto",
        zorder=0,
    )

    tc = tripcolor(u, axes=ax, shading="gouraud", cmap="viridis", alpha=0.6)
    triang = mtri.Triangulation(
        mesh.coordinates.dat.data_ro[:, 0],
        mesh.coordinates.dat.data_ro[:, 1],
        mesh.coordinates.cell_node_map().values,
    )
    ax.triplot(triang, linewidth=0.1, color="k", zorder=2)
    fig.colorbar(tc, ax=ax, label=str(da.name))
    ax.set_title(f"Adapted mesh at time={str(slice_t_na['time'].values)}")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_xlim(na_lon_min, na_lon_max)
    ax.set_ylim(na_lat_min, na_lat_max)
    fig.tight_layout()

    frame_path = os.path.join(frames_dir, f"frame_{t_idx}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    frame_paths.append(frame_path)

    mesh.coordinates.assign(reg_coords)

gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "weather_mesh_adapted_85710_85720.gif"))
frames = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(gif_path, frames, duration=0.5)
print("Saved GIF:", gif_path)

