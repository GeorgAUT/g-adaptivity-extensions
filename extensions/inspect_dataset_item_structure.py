import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_mixed import MeshInMemoryDataset_Mixed
from data import MeshInMemoryDataset


# Pick one of the existing processed datasets shipped in this repo
root_test_mixed = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "data",
        "Poisson_randg_mix_2d_test_25_monitor_hessian_rectangle_15_0.1reg_2_3gauss_iso_bin",
    )
)

root_train_mixed = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "data",
        "Poisson_randg_mix_2d_train_100_monitor_hessian_rectangle_15_0.1reg_2_3gauss_iso_bin",
    )
)

# Minimal opt needed for dataset init + loading. The dataset will load real attributes from disk.
opt = {
    "pde_type": "Poisson",
    "data_type": "randg_mix",
    "mesh_geometry": "rectangle",
    "mesh_file_type": "bin",
    "eval_quad_points": 101,
    "eval_refinement_level": 2,
    "HO_degree": 4,
    "monitor_type": "monitor_hessian",
    "mon_reg": 0.1,
    "mon_power": 0.2,
    "mesh_scale": 1.0,
    "anis_gauss": False,
    "num_gauss": 2,
    "num_gauss_range": [2, 3],
    "use_mpi_optimized": False,
    "mesh_dims_train": [[15, 15], [20, 20]],
    "mesh_dims_test": [[i, i] for i in range(12, 24, 1)],
}

print("--- Dataset roots ---")
print("root_test_mixed:", root_test_mixed)
print("root_train_mixed:", root_train_mixed)

print("\n--- Loading dataset (mixed / test) ---")

dataset = MeshInMemoryDataset_Mixed(
    root=root_test_mixed,
    num_data=25,
    mesh_dims=[15, 15],
    train_test="test",
    opt=opt,
)

# IMPORTANT:
# Do NOT call dataset[0] / dataset.get(0) here.
# For the mixed dataset class, __getitem__ triggers get(), which reloads Firedrake Functions
# from PETSc binary files on disk. In this repo snapshot those binaries can be inconsistent
# with the mesh inferred from the stored PyG Data object, leading to VecLoad size errors.
#
# For schema inspection we only need the already-materialized PyG Data stored in data_list.
item = dataset.data_list[0]

print("\n--- data_list[0] context ---")
print("num_nodes:", getattr(item, "num_nodes", None))
if getattr(item, "num_nodes", None) is not None:
    n_meshpoints = int((item.num_nodes) ** 0.5)
    print("inferred n_meshpoints (sqrt(num_nodes)):", n_meshpoints)

print("\n--- Item type ---")
print(type(item))

print("\n--- Item keys (PyG storage keys) ---")
try:
    print(sorted(item.keys()))
except Exception as e:
    print("Could not list keys via item.keys():", repr(e))

print("\n--- Attribute existence checks (things Mesh_Adaptor may access) ---")
required_like = [
    "x_in",
    "edge_index",
    "coarse_mesh",
    "u_coarse_reg",
    "Hessian_Frob_u_tensor",
    "patch_indices",
    "boundary_nodes",
    "boundary_nodes_dict",
    "node_boundary_map",
]

for k in required_like:
    try:
        v = getattr(item, k)
        if isinstance(v, torch.Tensor):
            print(f"{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            inner0 = v[0] if len(v) else None
            print(f"{k}: {type(v).__name__} len={len(v)} inner0_type={type(inner0)}")
        else:
            print(f"{k}: {type(v)}")
    except Exception as e:
        print(f"{k}: MISSING ({type(e).__name__}: {e})")

print("\n--- A few shapes/values ---")
if hasattr(item, "x_in"):
    print("x_in[0:3] =", item.x_in[:3])
if hasattr(item, "edge_index"):
    print("edge_index shape =", tuple(item.edge_index.shape))

print("\n--- Notes ---")
print(
    "If your demo crashes with missing Hessian_Frob_u_tensor, it means your constructed Data object\n"
    "does not include that field while your opt['new_model_monitor_type'] is set to 'Hessian_Frob_u_tensor'.\n"
    "If you switch opt['new_model_monitor_type'] to 'UM2N', the model instead reads u_coarse_reg[0]."
)
