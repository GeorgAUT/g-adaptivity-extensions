import sys
import os
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
# import matplotlib
# matplotlib.use('TkAgg')  # Switch to a different backend to avoid PyCharm's custom backend issues
import matplotlib.pyplot as plt

from firedrake.pyplot import tripcolor
from firedrake import *
from firedrake import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, DirichletBC, CheckpointFile

from classical_meshing.ma_mesh_1d import deform_mesh_mmpde1d
from classical_meshing.ma_mesh_2d import deform_mesh_ma2d
from classical_meshing.ma_mesh_3d import deform_mesh_ma3d

from params import get_params, run_params
from utils_data import make_data_name, to_float32, convert_to_boundary_mask, \
    map_firedrake_to_cannonical_ordering_2d, map_firedrake_to_cannonical_ordering_1d, \
    map_firedrake_to_cannonical_ordering_3d, save_function, load_function, load_vector_function
from pde_solvers import get_solve_firedrake_class


class PyG_Dataset(object):
  def __init__(self, data):
    self.data = data
    self.num_nodes = data.x_comp.shape[0]
    self.num_node_features = data.x_comp.shape[1]

class MeshInMemoryDataset(pyg.data.InMemoryDataset):
    def __init__(self, root, train_test, num_data, mesh_dims, opt, transform=None, pre_transform=None):
        self.root = root
        self.train_test = train_test
        self.num_data = num_data
        self.opt = opt
        self.dim = len(mesh_dims)

        if self.dim == 1:
            self.n = mesh_dims[0]
        elif self.dim == 2:
            self.n = mesh_dims[0]
            self.m = mesh_dims[1]
        elif self.dim == 3:
            self.n = mesh_dims[0]
            self.m = mesh_dims[1]
            self.l = mesh_dims[2]

        # Initialize patch configuration if enabled
        self.use_patches = opt.get('use_patches', False)
        if self.use_patches and self.dim == 2:
            self.num_patches_x = opt.get('num_patches_x', 10)
            self.num_patches_y = opt.get('num_patches_y', 10)
            self.patch_size_x = 1.0 / self.num_patches_x
            self.patch_size_y = 1.0 / self.num_patches_y

        self.num_x_comp_features = self.dim
        self.num_x_ma_features = self.dim
        self.x_coarse_shared = None
        self.x_fine_shared = None
        self.mapping_dict = None
        self.mapping_tensor = None
        self.mapping_dict_fine = None
        self.mapping_tensor_fine = None

        super(MeshInMemoryDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        print("Dataset loaded with", len(self.data), "items.")

        custom_attributes_path = os.path.join(self.root, "processed", "custom_attributes.pt")
        if os.path.exists(custom_attributes_path):
            if self.opt['mesh_geometry'] == 'rectangle':
                custom_attributes = torch.load(custom_attributes_path)
                self.x_coarse_shared = custom_attributes['x_coarse_shared']
                self.x_fine_shared = custom_attributes['x_fine_shared']
                self.mapping_tensor = custom_attributes['mapping_tensor']
                self.mapping_dict = custom_attributes['mapping_dict']
                self.mapping_tensor_fine = custom_attributes['mapping_tensor_fine']
                self.mapping_dict_fine = custom_attributes['mapping_dict_fine']
                self.orig_opt = custom_attributes['orig_opt']
            else:
                Warning(f"Mapping to canonical ordering not implemented for unstructured meshes")
                custom_attributes = torch.load(custom_attributes_path)
                self.x_coarse_shared = custom_attributes['x_coarse_shared']
                self.x_fine_shared = custom_attributes['x_fine_shared']
                self.orig_opt = custom_attributes['orig_opt']

        # Load the meshes
        if opt['mesh_file_type'] == 'h5':
            # Load from HDF5 CheckpointFile
            processed_dir = os.path.join(self.root, "processed")
            os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists
            coarse_mesh_file_path = os.path.join(processed_dir, "coarse_mesh.h5")
            fine_mesh_file_path = os.path.join(processed_dir, "fine_mesh.h5")
            if os.path.exists(coarse_mesh_file_path):
                with CheckpointFile(coarse_mesh_file_path, 'r') as mesh_file:
                    self.coarse_mesh = mesh_file.load_mesh("coarse_mesh")
            if os.path.exists(fine_mesh_file_path):
                with CheckpointFile(fine_mesh_file_path, 'r') as mesh_file:
                    self.fine_mesh = mesh_file.load_mesh("fine_mesh")
        elif opt['mesh_file_type'] == 'bin':
            self.init_mesh()

        SolverCoarse = get_solve_firedrake_class(opt)
        self.PDESolver_coarse = SolverCoarse(opt, self.dim, self.coarse_mesh)
        SolverFine = get_solve_firedrake_class(opt)
        self.PDESolver_fine = SolverFine(opt, self.dim, self.fine_mesh)

        #Hack for NavierStokes subset testing
        if opt['data_type'] == "RBF":
            data_idxs = opt['train_idxs'] if self.train_test == 'train' else opt['test_idxs']
            self.data_idx_dict = dict(zip(range(len(data_idxs)), data_idxs))


    def init_mesh(self):
        # Create PDE solvers
        opt = self.opt

        if opt['mesh_geometry'] == 'rectangle':
            if self.dim == 1:
                self.coarse_mesh = UnitIntervalMesh(self.n - 1, name="coarse_mesh")
                self.fine_mesh = UnitIntervalMesh(opt['eval_quad_points'] - 1, name="fine_mesh")
            elif self.dim == 2:
                self.coarse_mesh = UnitSquareMesh(self.n - 1, self.m - 1, name="coarse_mesh")
                self.fine_mesh = UnitSquareMesh(opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, name="fine_mesh")
            elif self.dim == 3:
                self.coarse_mesh = UnitCubeMesh(self.n - 1, self.m - 1, self.l - 1, name="coarse_mesh")
                self.fine_mesh = UnitCubeMesh(opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, opt['eval_quad_points'] - 1, name="fine_mesh")

            # Store initial coordinates
            self.initial_coarse_coords = self.coarse_mesh.coordinates.copy(deepcopy=True)
            self.initial_fine_coords = self.fine_mesh.coordinates.copy(deepcopy=True)

        else:
            try:
                self.coarse_mesh = Mesh(f"../meshes/{opt['mesh_geometry']}.msh")
                self.coarse_mesh.name = "coarse_mesh"

                # Create a fine mesh using MeshHierarchy
                mesh_hierarchy = MeshHierarchy(self.coarse_mesh, opt['eval_refinement_level'])
                self.fine_mesh = mesh_hierarchy[-1]
                self.fine_mesh.name = "fine_mesh"
                
                # Store initial coordinates
                self.initial_coarse_coords = self.coarse_mesh.coordinates.copy(deepcopy=True)
                self.initial_fine_coords = self.fine_mesh.coordinates.copy(deepcopy=True)
            except:
                raise ValueError(f"Unknown mesh geometry: {opt['mesh_geometry']}")
                sys.exit()

    def reset_mesh_coordinates(self):
        """Reset mesh coordinates to their initial state"""
        if hasattr(self, 'initial_coarse_coords'):
            self.coarse_mesh.coordinates.assign(self.initial_coarse_coords)
        if hasattr(self, 'initial_fine_coords'):
            self.fine_mesh.coordinates.assign(self.initial_fine_coords)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        opt = self.opt

        # First initialize the mesh regardless of file type
        self.init_mesh()

        # Save the meshes to HDF5 files if needed
        if opt['mesh_file_type'] == 'h5':
            processed_dir = os.path.join(self.root, "processed")
            os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists
            with CheckpointFile(os.path.join(processed_dir, "coarse_mesh.h5"), 'w') as mesh_file:
                mesh_file.save_mesh(self.coarse_mesh)
            with CheckpointFile(os.path.join(processed_dir, "fine_mesh.h5"), 'w') as mesh_file:
                mesh_file.save_mesh(self.fine_mesh)

        SolverCoarse = get_solve_firedrake_class(opt)
        self.PDESolver_coarse = SolverCoarse(opt, self.dim, self.coarse_mesh)
        SolverFine = get_solve_firedrake_class(opt)
        self.PDESolver_fine = SolverFine(opt, self.dim, self.fine_mesh)

        self.x_coarse_shared = torch.tensor(self.coarse_mesh.coordinates.dat.data_ro)
        self.x_fine_shared = torch.tensor(self.fine_mesh.coordinates.dat.data_ro)

        # Map fd meshes to canonical ordering
        if opt['mesh_geometry'] == 'rectangle':
            if self.dim == 1:
                mapping_dict, mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(self.x_coarse_shared, self.n)
                mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, X_fd_vec_fine = map_firedrake_to_cannonical_ordering_1d(self.x_fine_shared, self.opt['eval_quad_points'])
            elif self.dim == 2:
                mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(self.x_coarse_shared, self.n, self.m)
                mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine = map_firedrake_to_cannonical_ordering_2d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'])
            elif self.dim == 3:
                mapping_dict, mapping_tensor, X_fd_grid, Y_fd_grid, Z_fd_grid, X_fd_vec, Y_fd_vec, Z_fd_vec = map_firedrake_to_cannonical_ordering_3d(self.x_coarse_shared, self.n, self.m, self.l)
                mapping_dict_fine, mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, Z_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine, Z_fd_vec_fine = map_firedrake_to_cannonical_ordering_3d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'], self.opt['eval_quad_points'])

            self.mapping_dict = mapping_dict
            self.mapping_tensor = mapping_tensor
            self.mapping_dict_fine = mapping_dict_fine
            self.mapping_tensor_fine = mapping_tensor_fine

            custom_attributes = {
                'x_coarse_shared': self.x_coarse_shared,
                'x_fine_shared': self.x_fine_shared,
                'mapping_dict': self.mapping_dict,
                'mapping_tensor': self.mapping_tensor,
                'mapping_dict_fine': self.mapping_dict_fine,
                'mapping_tensor_fine': self.mapping_tensor_fine,
                'orig_opt': opt.as_dict() if opt['wandb'] else opt
            }
        else:
            Warning(f"Mapping to canonical ordering not implemented for unstructured meshes")
            custom_attributes = {
                'x_coarse_shared': self.x_coarse_shared,
                'x_fine_shared': self.x_fine_shared,
                'orig_opt': opt.as_dict() if opt['wandb'] else opt
            }

        torch.save(custom_attributes, os.path.join(self.root, "processed", "custom_attributes.pt"))

        data_list = []
        if opt['data_type'] in ['structured']:
            num_data_dict = {1: 9, 2: 25}
            self.num_data = num_data_dict[self.dimself.dim]
            data_idxs = range(self.num_data)
        elif opt['data_type'] == "RBF":
            data_idxs = opt['train_idxs'] if self.train_test == 'train' else opt['test_idxs']
            self.data_idx_dict = dict(zip(range(len(data_idxs)), data_idxs))
        else:
            data_idxs = range(self.num_data)

        for idx in data_idxs:
            data = firedrake_mesh_to_PyG(self.coarse_mesh)

            # Get PDE specific parameters
            pde_params = self.PDESolver_coarse.get_pde_params(idx, self.num_data, self.opt['num_gauss'])

            # Pass some global parameters from opt to pde_params for the solvers
            pde_params['mon_power'] = opt['mon_power']
            pde_params['monitor_type'] = opt['monitor_type']
            pde_params['mon_reg'] = opt['mon_reg']
            pde_params['num_gauss'] = opt['num_gauss']
            # Store pde_params in the data object
            data.pde_params = pde_params

            #sample data, update solver and solve PDE
            self.PDESolver_coarse.update_solver(pde_params)

            if opt['pde_type'] in ['Poisson','Burgers']:
                u = self.PDESolver_coarse.solve()
            elif opt['pde_type'] == 'NavierStokes':
                u, p = self.PDESolver_coarse.solve()

            #save inputs to GNN
            if opt['pde_type'] in ['Poisson','Burgers']:
                data.u_tensor = torch.from_numpy(u.dat.data) # FEM solution values
            elif opt['pde_type'] == 'NavierStokes':
                data.u_tensor = torch.from_numpy(u.dat.data) # FEM solution values
                data.p_tensor = torch.from_numpy(p.dat.data)

            Hessian_Frob_u = self.PDESolver_coarse.get_Hessian_Frob_norm()
            data.Hessian_Frob_u_tensor = torch.from_numpy(Hessian_Frob_u.dat.data)

            # For Poisson save the exact f
            if opt['pde_type'] == 'Poisson':
                pde_data = self.PDESolver_coarse.get_pde_data(pde_params)
                pde_fs = self.PDESolver_coarse.get_pde_function_space()
                f_data = project(pde_data['f'], pde_fs)
                data.f_tensor = torch.from_numpy(f_data.dat.data)

            self.PDESolver_fine.update_solver(pde_params)
            if opt['pde_type'] in ['Poisson','Burgers']:
                uu_ref = self.PDESolver_fine.solve()
            elif opt['pde_type'] == 'NavierStokes':
                uu_ref, pp_ref = self.PDESolver_fine.solve()

            #Deform mesh using MMPDE/MA
            if opt['pde_type'] in ['Poisson','Burgers']:
                if self.dim == 1:
                    data.x_ma, data.ma_its, data.build_time = deform_mesh_mmpde1d(self.x_coarse_shared, self.n, pde_params)
                elif self.dim == 2:
                        x_ma, data.ma_its, data.build_time = deform_mesh_ma2d(self.x_coarse_shared,self.coarse_mesh, self.PDESolver_coarse, u, Hessian_Frob_u, opt, pde_params, SolverCoarse)
                        data.x_ma = torch.from_numpy(x_ma)
                elif self.dim == 3:
                    x_ma, data.ma_its, data.build_time = deform_mesh_ma3d(self.x_coarse_shared, self.coarse_mesh, self.PDESolver_coarse, u, Hessian_Frob_u, opt, pde_params, SolverCoarse)
                    data.x_ma = torch.from_numpy(x_ma)

            elif opt['pde_type'] == 'NavierStokes':
                data.x_ma = data.x_comp
                #store dummy values for MA as N/A
                data.ma_its = 1
                data.build_time = 0.0

            # Build suffix for PDE data
            if opt['data_type'] in ['randg']:
                filename_suffix_coarse = f"dim_{self.dim}_mon_{pde_params['mon_power']}_reg_{pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data_coarse.{opt['mesh_file_type']}"
                filename_suffix_fine = f"dim_{self.dim}_mon_{pde_params['mon_power']}_reg_{pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data_fine.{opt['mesh_file_type']}"

            elif opt['data_type'] in ['structured']:
                scale_val = round(pde_params['scale_value'], 2)
                filename_suffix_coarse = f"dim_{self.dim}_scale_{scale_val}_mon_{pde_params['mon_power']}_reg_{pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data_coarse.{opt['mesh_file_type']}"
                filename_suffix_fine = f"dim_{self.dim}_scale_{scale_val}_mon_{pde_params['mon_power']}_reg_{pde_params['mon_reg']}_{opt['num_gauss']}gauss_{idx}_pde_data_fine.{opt['mesh_file_type']}"

            elif opt['data_type'] in ['RBF'] and opt['pde_type'] == 'NavierStokes':
                filename_suffix_coarse = f"dim_{self.dim}_{idx}_pde_data_coarse.{opt['mesh_file_type']}"
                filename_suffix_fine = f"dim_{self.dim}_{idx}_pde_data_fine.{opt['mesh_file_type']}"
                filename_suffix_coarse_p = f"dim_{self.dim}_{idx}_pde_data_coarse_p.{opt['mesh_file_type']}"
                filename_suffix_fine_p = f"dim_{self.dim}_{idx}_pde_data_fine_p.{opt['mesh_file_type']}"

            pde_data_file_coarse = os.path.join(self.root, "processed", filename_suffix_coarse)
            pde_data_file_fine = os.path.join(self.root, "processed", filename_suffix_fine)

            if opt['pde_type'] == 'NavierStokes':
                pde_data_file_coarse_p = os.path.join(self.root, "processed", filename_suffix_coarse_p)
                pde_data_file_fine_p = os.path.join(self.root, "processed", filename_suffix_fine_p)

            # ----- Save PDE solution + mesh if desired -----
            # HDF5 or BIN
            if opt['mesh_file_type'] == 'h5':
                with CheckpointFile(pde_data_file_coarse, 'w') as pde_file:
                    pde_file.save_mesh(self.coarse_mesh)
                    if opt['pde_type'] in ['Poisson', 'Burgers']:
                        pde_file.save_function(u, name="u_coarse")
                    elif opt['pde_type'] == 'NavierStokes':
                        pde_file.save_function(u, name="u_coarse")
                        pde_file.save_function(p, name="p_coarse")
                with CheckpointFile(pde_data_file_fine, 'w') as pde_file:
                    pde_file.save_mesh(self.fine_mesh)
                    if opt['pde_type'] in ['Poisson', 'Burgers']:
                        pde_file.save_function(uu_ref, name="uu_ref")
                    elif opt['pde_type'] == 'NavierStokes':
                        pde_file.save_function(uu_ref, name="uu_ref")
                        pde_file.save_function(pp_ref, name="pp_ref")

            elif opt['mesh_file_type'] == 'bin':
                if opt['pde_type'] in ['Poisson', 'Burgers']:
                    save_function(u, pde_data_file_coarse)
                    save_function(uu_ref, pde_data_file_fine)
                elif opt['pde_type'] == 'NavierStokes':
                    save_function(u, pde_data_file_coarse)
                    save_function(p, pde_data_file_coarse_p)
                    save_function(uu_ref, pde_data_file_fine)
                    save_function(pp_ref, pde_data_file_fine_p)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        data.apply(to_float32)
        torch.save((data, slices), self.processed_paths[0])


    def get(self, idx):
        opt = self.opt
        data = super().get(idx)
        data.x_coarse = self.x_coarse_shared.float()
        data.x_fine = self.x_fine_shared.float()
        if isinstance(data.x_ma, np.ndarray):
            data.x_ma = torch.from_numpy(data.x_ma)

        mon_power = round(data.pde_params['mon_power'].item(), 2)
        num_gauss = data.pde_params['num_gauss'].item()
        if 'mon_reg' in data.pde_params:
            mon_reg = round(data.pde_params['mon_reg'].item(), 2)

        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            if opt['data_type'] in ['randg']:
                filename_suffix_coarse = (
                    f"dim_{self.dim}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data_coarse.{opt['mesh_file_type']}"
                )
                filename_suffix_fine = (
                    f"dim_{self.dim}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data_fine.{opt['mesh_file_type']}"
                )
            else:
                scale_val = round(data.pde_params['scale_value'].item(), 2)
                filename_suffix_coarse = (
                    f"dim_{self.dim}_scale_{scale_val}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data_coarse.{opt['mesh_file_type']}"
                )
                filename_suffix_fine = (
                    f"dim_{self.dim}_scale_{scale_val}_mon_{mon_power}_reg_{mon_reg}_{num_gauss}gauss_{idx}_pde_data_fine.{opt['mesh_file_type']}"
                )
        elif opt['data_type'] in ['RBF'] and opt['pde_type'] == 'NavierStokes':
            file_idx = self.data_idx_dict[idx]
            filename_suffix_coarse = f"dim_{self.dim}_{file_idx}_pde_data_coarse.{opt['mesh_file_type']}"
            filename_suffix_fine = f"dim_{self.dim}_{file_idx}_pde_data_fine.{opt['mesh_file_type']}"
            filename_suffix_coarse_p = f"dim_{self.dim}_{file_idx}_pde_data_coarse_p.{opt['mesh_file_type']}"
            filename_suffix_fine_p = f"dim_{self.dim}_{file_idx}_pde_data_fine_p.{opt['mesh_file_type']}"

        pde_data_file_coarse = os.path.join(self.root, "processed", filename_suffix_coarse)
        pde_data_file_fine = os.path.join(self.root, "processed", filename_suffix_fine)

        if opt['pde_type'] == 'NavierStokes':
            pde_data_file_coarse_p = os.path.join(self.root, "processed", filename_suffix_coarse_p)
            pde_data_file_fine_p = os.path.join(self.root, "processed", filename_suffix_fine_p)

        if opt['mesh_file_type'] == 'h5':
            # --- HDF5 loading ---
            with CheckpointFile(pde_data_file_coarse, 'r') as pde_file:
                self.coarse_mesh = pde_file.load_mesh("coarse_mesh")
                u_coarse_reg = pde_file.load_function(self.coarse_mesh, "u_coarse")
                if opt['pde_type'] == 'NavierStokes':
                    p_coarse_reg = pde_file.load_function(self.coarse_mesh, "p_coarse")

            with CheckpointFile(pde_data_file_fine, 'r') as pde_file:
                self.fine_mesh = pde_file.load_mesh("fine_mesh")
                uu_ref = pde_file.load_function(self.fine_mesh, "uu_ref")
                if opt['pde_type'] == 'NavierStokes':
                    pp_ref = pde_file.load_function(self.fine_mesh, "pp_ref")

        elif opt['mesh_file_type'] == 'bin':
            if opt['pde_type'] == 'Poisson':
                u_coarse_reg = load_function(self.coarse_mesh, pde_data_file_coarse, family = "CG", degree = 1)
                uu_ref = load_function(self.fine_mesh, pde_data_file_fine, family = "CG", degree = 1)
            elif opt['pde_type'] == 'Burgers':
                u_coarse_reg = load_vector_function(self.coarse_mesh, pde_data_file_coarse, family="CG", degree=1)
                uu_ref = load_vector_function(self.fine_mesh, pde_data_file_fine, family="CG", degree=1)
            elif opt['pde_type'] == 'NavierStokes':
                u_coarse_reg = load_vector_function(self.coarse_mesh, pde_data_file_coarse, family="CG", degree=2)
                uu_ref = load_vector_function(self.fine_mesh, pde_data_file_fine, family="CG", degree=2)
                p_coarse_reg = load_function(self.coarse_mesh, pde_data_file_coarse_p, family="CG", degree=1)
                pp_ref = load_function(self.fine_mesh, pde_data_file_fine_p, family="CG", degree=1)

        data.uu_ref = uu_ref
        data.u_coarse_reg = u_coarse_reg
        data.coarse_mesh = self.coarse_mesh

        if opt['pde_type'] == 'NavierStokes':
            data.p_coarse_reg = p_coarse_reg
            data.pp_ref = pp_ref

        return data

    # TODO this could be precomputed
    def get_patch_indices(self, node_coords):
        """Get indices of nodes in each patch
        Args:
            node_coords: [N, 2] array of node coordinates
        Returns:
            list of arrays containing node indices for each patch
        """
        if not self.use_patches:
            return None

        tol = 1e-8
        patches = []

        # Get domain bounds from actual node coordinates
        x_min_domain = node_coords[:, 0].min()
        x_max_domain = node_coords[:, 0].max()
        y_min_domain = node_coords[:, 1].min()
        y_max_domain = node_coords[:, 1].max()

        # Calculate patch sizes based on domain size
        patch_size_x = (x_max_domain - x_min_domain) / self.num_patches_x
        patch_size_y = (y_max_domain - y_min_domain) / self.num_patches_y

        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                # Calculate patch boundaries using actual domain bounds
                x_min = x_min_domain + i * patch_size_x
                x_max = x_min + patch_size_x
                y_min = y_min_domain + j * patch_size_y
                y_max = y_min + patch_size_y

                # Find nodes in this patch
                indices = np.where(
                    (node_coords[:, 0] >= x_min - tol) &
                    (node_coords[:, 0] <= x_max + tol) &
                    (node_coords[:, 1] >= y_min - tol) &
                    (node_coords[:, 1] <= y_max + tol)
                )[0]

                patches.append(indices)

        return patches


#generate a regular n x m mesh as a PyG data - not used anymore instead load from Firedrake
def generate_mesh_2d(n, m):
    ''' n for x axis, m for y axis '''
    x_points = torch.linspace(0, 1, n + 1)
    y_points = torch.linspace(0, 1, m + 1)
    x, y = torch.meshgrid(x_points, y_points)
    x = x.reshape(-1)
    y = y.reshape(-1)
    points = torch.stack([x, y], dim=1)
    edges = []

    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m:
                edges.append([i * (m + 1) + j, i * (m + 1) + j + 1])
                edges.append([i * (m + 1) + j, (i + 1) * (m + 1) + j])
            elif i < n:
                edges.append([i * (m + 1) + j, (i + 1) * (m + 1) + j])
            elif j < m:
                edges.append([i * (m + 1) + j, i * (m + 1) + j + 1])
    edges = torch.tensor(edges).T

    #make edges undirected
    edges = torch.cat([edges, torch.flip(edges, dims=[0])], dim=1)

    boundary_nodes = set()
    for i in range(n + 1):
        boundary_nodes.add(i) #bottom row
        boundary_nodes.add(m * (n + 1) + i) #top row
    for i in range(m + 1):
        boundary_nodes.add(i * (m + 1))  # left column
        boundary_nodes.add(n + i * (n + 1)) # right column

    boundary_nodes = list(boundary_nodes)

    boundary_nodes = convert_to_boundary_mask(boundary_nodes, num_nodes=(n + 1) * (m + 1))

    return PyG_Dataset(pyg.data.Data(x_comp=points, x_ma=points,
                                          edge_index=edges, n=n, m=m, num_node_features=2, boundary_nodes=boundary_nodes))


def firedrake_mesh_to_PyG(mesh):
    # Get coordinates of the vertices
    coordinates = mesh.coordinates.dat.data_ro
    # Get the cell to node mapping
    cell_node_map = mesh.coordinates.cell_node_map().values
    # Initialize a set for edges (each edge represented as a tuple)
    edges_set = set()
    # Iterate through each cell
    for cell in cell_node_map:
        # For each pair of nodes in the cell, add an edge
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                # Add edge in both directions to ensure it's undirected
                edges_set.add((cell[i], cell[j]))
                edges_set.add((cell[j], cell[i]))

    # Convert edge set to a tensor
    edge_index = torch.tensor(list(edges_set), dtype=torch.long).t().contiguous()
    # Define a function space on the mesh
    V = FunctionSpace(mesh, "CG", 1)
    # Create a boundary condition
    bc = DirichletBC(V, 0, "on_boundary")
    # Get the boundary nodes
    boundary_nodes = bc.nodes

    boundary_nodes_mask = convert_to_boundary_mask(boundary_nodes, num_nodes=len(coordinates))

    boundary_nodes_dict = {}
    unique_boundary_ids = mesh.topology.exterior_facets.unique_markers
    for boundary_id in unique_boundary_ids:
        bc = DirichletBC(V, 0, boundary_id)
        boundary_nodes_dict[boundary_id] = bc.nodes

    boundary_ids = list(unique_boundary_ids)
    all_boundary_nodes = []
    for bid in boundary_ids:
        all_boundary_nodes.extend(V.boundary_nodes(bid))
    unique_nodes, counts = np.unique(all_boundary_nodes, return_counts=True)
    corner_nodes = unique_nodes[counts > 1]

    #mask for edges who's dst node is in the boundary and source node is in the interior
    to_boundary_edge_mask = torch.tensor([edge_index[1][i].item() in boundary_nodes and edge_index[0][i].item() not in boundary_nodes for i in range(edge_index.shape[1])])
    #mask for edges who's dst node is in the corner
    # to_corner_nodes_mask = torch.tensor([edge_index[1][i].item() in corner_nodes and edge_index[0][i].item() not in boundary_nodes for i in range(edge_index.shape[1])])
    to_corner_nodes_mask = torch.tensor([edge_index[1][i].item() in corner_nodes for i in range(edge_index.shape[1])])

    # Invert the boundary_nodes_dict to get node:boundary_id mapping
    node_boundary_map = {}
    for boundary_id, nodes in boundary_nodes_dict.items():
        for node in nodes:
            if node not in node_boundary_map.keys():
                node_boundary_map[node] = [boundary_id]
            else:
                node_boundary_map[node].append(boundary_id)

    # Create a mask for edges between different boundaries, excluding corner nodes
    diff_boundary_edges_mask_list = []
    for edge in edge_index.t().tolist():
        src_node, dst_node = edge
        if src_node in node_boundary_map.keys() and dst_node in node_boundary_map.keys():
            src_boundary_list = node_boundary_map[src_node]
            dst_boundary_list = node_boundary_map[dst_node]
            # Check if both nodes in edge are on different boundaries and neither is a corner node
            if src_boundary_list != dst_boundary_list and src_node not in corner_nodes and dst_node not in corner_nodes:
                diff_boundary_edges_mask_list.append(True)
            else:
                diff_boundary_edges_mask_list.append(False)
        else:
            diff_boundary_edges_mask_list.append(False)

    diff_boundary_edges_mask = torch.tensor(diff_boundary_edges_mask_list)

    # Create the PyG graph
    data = pyg.data.Data(x_comp=torch.tensor(coordinates, dtype=torch.float),
                            x_in=torch.tensor(coordinates, dtype=torch.float),
                            x_ma=torch.tensor(coordinates, dtype=torch.float), edge_index=edge_index,
                            boundary_nodes=boundary_nodes_mask, corner_nodes=corner_nodes, to_boundary_edge_mask=to_boundary_edge_mask,
                            to_corner_nodes_mask=to_corner_nodes_mask, diff_boundary_edges_mask=diff_boundary_edges_mask,
                            boundary_nodes_dict=boundary_nodes_dict, node_boundary_map=node_boundary_map)
    return data


def plot_initial_dataset_1d(dataset, opt, plot_reg_mesh=True, plot_ma_mesh=True):
    """
    Plot the initial dataset for 1D problems using precomputed solution data.
    
    Args:
        dataset: The dataset containing the PDE solutions
        opt: Options dictionary containing parameters
        plot_reg_mesh: Whether to plot solutions on the regular mesh
        plot_ma_mesh: Whether to plot solutions on the MA mesh
    """

    title_suffix = f"{opt['data_name']}"
    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Set up plot configurations
    max_plots = 9
    nrows, ncols = 3, 3
    
    # Create figures based on plot flags
    figures = {}
    axes = {}
    
    if plot_reg_mesh:
        figures['reg'] = plt.figure(figsize=(15, 15))
        figures['reg'].suptitle(f'Regular mesh solutions - {title_suffix}', fontsize=20)
        axes['reg'] = [figures['reg'].add_subplot(nrows, ncols, i+1) for i in range(max_plots)]
    
    if plot_ma_mesh:
        figures['ma'] = plt.figure(figsize=(15, 15))
        figures['ma'].suptitle(f'MA mesh solutions - {title_suffix}', fontsize=20)
        axes['ma'] = [figures['ma'].add_subplot(nrows, ncols, i+1) for i in range(max_plots)]
    
    # Loop over the dataset
    for i, data in enumerate(loader):
        if i >= max_plots:
            break
        
        # Plot on regular mesh
        if plot_reg_mesh:
            ax = axes['reg'][i]
            
            # Plot precomputed solution using stored data
            x_comp = data.x_comp.cpu().numpy()
            u_true_val = data.u_true[0].cpu().numpy()
            u_pred_val = data.uu[0].cpu().numpy()
            
            # Plot solutions
            ax.plot(x_comp, u_pred_val, label='FEM Solution', color='orange')
            ax.plot(x_comp, u_true_val, label='True Solution', color='green')
            
            # Add mesh tick marks to show discretization
            ymin, ymax = ax.get_ylim()
            dash_length = 0.04 * (ymax - ymin)
            ymin_dash = ymin - 0.02 * (ymax - ymin)
            
            for x_pos in x_comp:
                ax.plot([x_pos, x_pos], [ymin_dash, ymin_dash + dash_length], 
                        color='black', linestyle='-', linewidth=1)
            
            ax.set_title(f'Sample {i} - Regular Mesh')
            ax.legend()
            
        # Plot on MA mesh
        if plot_ma_mesh:
            ax = axes['ma'][i]
            
            # Plot precomputed MA solution
            x_phys = data.x_phys.cpu().numpy()
            
            # Display MA mesh points
            for x_pos in x_phys:
                ymin, ymax = ax.get_ylim() if ax.get_ylim()[1] > ax.get_ylim()[0] else (-0.1, 1.0)
                dash_length = 0.04 * (ymax - ymin)
                ymin_dash = ymin - 0.02 * (ymax - ymin)
                ax.plot([x_pos, x_pos], [ymin_dash, ymin_dash + dash_length], 
                        color='black', linestyle='-', linewidth=1)
            
            # Plot precomputed solution if available
            if hasattr(data, 'uu_ma') and data.uu_ma is not None:
                ax.plot(x_phys, data.uu_ma[0].cpu().numpy(), label='FEM Solution (MA)', color='orange')
                ax.plot(x_phys, data.u_true_ma[0].cpu().numpy(), label='True Solution (MA)', color='green')
            else:
                # Just show the mesh points if solution isn't available
                ax.scatter(x_phys, np.zeros_like(x_phys), color='blue', marker='x', 
                          label='MA Mesh Points')
                ax.set_ylim([-0.1, 1.0])  # Default view range if no data
            
            # For comparison, also show regular mesh points
            x_comp = data.x_comp.cpu().numpy()
            ax.scatter(x_comp, np.zeros_like(x_comp) - 0.05, color='red', marker='o', 
                      s=20, label='Regular Mesh Points')
            
            ax.set_title(f'Sample {i} - MA Mesh')
            ax.legend()
    
    # Apply layout and save figures
    for name, fig in figures.items():
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        fig.savefig(f"../data/{opt['data_name']}_{name}_mesh.pdf", format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_initial_dataset_2d(dataset, opt, plot_mesh=True, plot_fem0=True, plot_fem1p=False):
    title_suffix = f"{opt['data_name']}"
    # Create a DataLoader with batch size of 1 to load one data point at a time
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if plot_mesh:
        # figure for mesh
        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
        fig.suptitle(f"Monge Ampere mesh - {opt['mesh_dims']} regularisation {opt['mon_reg']}", fontsize=20)
        axs = axs.ravel()

    if plot_fem0:
        #figure for firedrake function
        fig2, axs2 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
        fig2.suptitle(f"FEM solution for Gaussian scales {opt['scale']}", fontsize=20)
        fig2.subplots_adjust(top=0.2)
        axs2 = axs2.ravel()

    if plot_fem1p:
        #figure for torch tensor
        fig3, axs3 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
        fig3.suptitle(f'FEM uu on mesh {title_suffix}', fontsize=20)
        axs3 = axs3.ravel()

    # Loop over the dataset
    for i, data in enumerate(loader):
        if i == 25:
            break
        if plot_mesh:
            # Convert PyG graph to NetworkX graph
            G = to_networkx(data, to_undirected=True)
            x = data.x_ma
            positions = {i: x[i].tolist() for i in range(x.shape[0])}
            nx.draw(G, pos=positions, ax=axs[i], node_size=1, width=0.5, with_labels=False)

        if plot_fem0:
            #plot the FireDrake function
            colors = tripcolor(data.uu_ref[0], axes=axs2[i])

        if plot_fem1p:
            #plot the torch tensor
            x_comp_cannon = data.x_comp[dataset.mapping_tensor].cpu().numpy()
            uu_cannon = data.uu_tensor[dataset.mapping_tensor].cpu().numpy()
            contourf = axs3[i].tricontourf(x_comp_cannon[:, 0], x_comp_cannon[:, 1], uu_cannon, levels=15, cmap='viridis')

    #saves figs
    if plot_mesh:
        fig.tight_layout()
        fig.savefig(f"../data/{opt['data_name']}_mesh.pdf", format='pdf', dpi=300, bbox_inches='tight')
    if plot_fem0:
        fig2.tight_layout()
        fig2.savefig(f"../data/{opt['data_name']}_uu.pdf", format='pdf', dpi=300, bbox_inches='tight')
    if plot_fem1p:
        fig3.tight_layout()
        fig3.savefig(f"../data/{opt['data_name']}_uu_torch.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def plot_initial_dataset_3d(dataset, opt, num_plots=25):
    """Plot 3D meshes from the dataset.

    Args:
        dataset: The dataset containing 3D mesh data
        opt: Options dictionary
        num_plots: Number of meshes to plot (default 25)
    """
    title_suffix = f"{opt['data_name']}"

    # Create a DataLoader with batch size of 1
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Set up the subplot grid (5x5 grid)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"3D Mesh - {opt['mesh_dims']} regularisation {opt['mon_reg']}", fontsize=20)

    # Loop over the dataset
    for i, data in enumerate(loader):
        if i == num_plots:
            break

        # Create subplot
        ax = fig.add_subplot(5, 5, i + 1, projection='3d')

        # Convert PyG graph to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Get node positions from the coordinates
        x = data.x_ma
        positions = {i: x[i].tolist() for i in range(x.shape[0])}

        # Get edge list and positions for plotting
        edge_list = list(G.edges())
        edge_pos = np.array([(positions[u], positions[v]) for u, v in edge_list])

        # Plot edges
        for edge in edge_pos:
            ax.plot3D(
                [edge[0][0], edge[1][0]],  # x coordinates
                [edge[0][1], edge[1][1]],  # y coordinates
                [edge[0][2], edge[1][2]],  # z coordinates
                'b-', linewidth=0.5, alpha=0.5
            )

        # Plot nodes
        node_pos = np.array(list(positions.values()))
        ax.scatter(
            node_pos[:, 0],
            node_pos[:, 1],
            node_pos[:, 2],
            c='r', s=1
        )

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Remove axis labels and ticks for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"../data/{opt['data_name']}_3d_mesh.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # mesh = UnitSquareMesh(10, 10) #note this is 10 squares in each direction, so 11 nodes in each direction
    # data = firedrake_mesh_to_PyG(mesh)
    # plot2d(data)    # Plot the graph
    # triplot(mesh)    # Plot the mesh

    # # Add labels for nodes
    # coords = mesh.coordinates.dat.data
    # for i, (x, y) in enumerate(coords):
    #     plt.text(x, y, str(i), color='red')  # Adjust the color, position, etc. as needed
    # plt.show()

    # # 3D cube box
    # mesh = BoxMesh(10, 10, 10, 1, 1, 1)
    # data = firedrake_mesh_to_PyG(mesh)
    # plot_3d_pyg_graph_interactive(data)

    opt = get_params()
    if not opt['wandb_sweep']:
        opt = run_params(opt)

    opt['pde_type'] = "Poisson" # "Poisson" or "Burgers"
    # opt['data_type'] = "structured" "structured"#"randg"
    opt['data_type'] = "randg" # "structured"#"randg
    opt['mesh_geometry'] = "polygon_010" # "rectangle" or "cube" or "cylinder"
    opt['monitor_type'] = "monitor_hessian" #"ma"  # ma or mmpde or M2N
    opt['num_train'] = 25
    opt['num_test'] = 25

    opt['dataset'] = f"fd_{opt['monitor_type']}_2d" #'fd_ma_grid_2d' #'fd_ma_L'#'fd_noisey_grid' #fd_ma_grid#'noisey_grid'
    opt['mesh_dims'] = [11, 11] #[15, 15] #[11, 11]
    opt['mon_reg'] = 0.1 #.1 #0.1 #0.01
    opt['anis_gauss'] = False
    opt['num_gauss_range'] = [2, 3]

    if opt['data_type'] == "structured":
        scales = [0.1, 0.2, 0.3]
        regs = [0.1, 1.]
        dims = [[11, 11], [15, 15]]

        for scale in scales:
            for reg in regs:
                for dim in dims:
                    opt['scale'] = scale
                    opt['mon_reg'] = reg
                    opt['mesh_dims'] = dim
                    for train_test in ['train', 'test']:
                        opt = make_data_name(opt, train_test)
                        if train_test == 'train':
                            dataset = MeshInMemoryDataset(f"{opt['data_dir']}/{opt['data_name']}", train_test, opt['num_train'], opt['mesh_dims'], opt)
                        elif train_test == 'test':
                            dataset = MeshInMemoryDataset(f"{opt['data_dir']}/{opt['data_name']}", train_test, opt['num_test'], opt['mesh_dims'], opt)

    elif opt['data_type'] == "randg":
        for train_test in ['train', 'test']:
            opt = make_data_name(opt, train_test)
            if train_test == 'train':
                dataset = MeshInMemoryDataset(f"{opt['data_dir']}/{opt['data_name']}", train_test, opt['num_train'], opt['mesh_dims'], opt)
            elif train_test == 'test':
                dataset = MeshInMemoryDataset(f"{opt['data_dir']}/{opt['data_name']}", train_test, opt['num_test'], opt['mesh_dims'], opt)

    # plot_initial_dataset_2d(dataset, opt)
