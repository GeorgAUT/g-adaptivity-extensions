from torch_geometric.data import Batch

from src.data import *
from src.data_mixed_loader import Mixed_DataLoader

class MeshInMemoryDataset_Mixed(pyg.data.InMemoryDataset):
    def __init__(self, root, num_data, mesh_dims, train_test, opt, transform=None, pre_transform=None, pre_filter=None):
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
            self.k = mesh_dims[2]

        # Initialize patch configuration if enabled
        self.use_patches = opt.get('use_patches', False)
        if self.use_patches and self.dim == 2:
            self.num_patches_x = opt.get('num_patches_x', 10)
            self.num_patches_y = opt.get('num_patches_y', 10)
            self.patch_size_x = 1.0 / self.num_patches_x
            self.patch_size_y = 1.0 / self.num_patches_y

        self.num_x_comp_features = self.dim
        self.num_x_ma_features = self.dim

        if self.train_test == "train":
            self.min_mesh_points = self.opt['mesh_dims_train'][0][0]
            self.max_mesh_points = self.opt['mesh_dims_train'][-1][0]
        elif self.train_test == "test":
            self.min_mesh_points = self.opt['mesh_dims_test'][0][0]
            self.max_mesh_points = self.opt['mesh_dims_test'][-1][0]

        super(MeshInMemoryDataset_Mixed, self).__init__(root, transform, pre_transform, pre_filter)

        self.data_list = torch.load(self.processed_paths[0], weights_only=False)  # Load the data list
        print("Dataset loaded with", len(self.data_list), "items.")

        custom_attributes_path = os.path.join(self.root, "processed", "custom_attributes.pt")
        if os.path.exists(custom_attributes_path):
            custom_attributes = torch.load(custom_attributes_path, weights_only=False)
            if opt['pde_type'] == 'Poisson':
                self.mapping_tensor_fine = custom_attributes['mapping_tensor_fine']
                self.mapping_dict_fine = custom_attributes['mapping_dict_fine']
            self.orig_opt = custom_attributes['orig_opt']

       # Load the meshes
        if opt['mesh_file_type'] == 'h5':
            # Load from HDF5 CheckpointFile
            processed_dir = os.path.join(self.root, "processed")
            os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists
            fine_mesh_file_path = os.path.join(processed_dir, "fine_mesh.h5")
            if os.path.exists(fine_mesh_file_path):
                with CheckpointFile(fine_mesh_file_path, 'r') as mesh_file:
                    self.fine_mesh = mesh_file.load_mesh("fine_mesh")
            for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
                coarse_mesh_file_path = os.path.join(processed_dir, "coarse_mesh.h5")
                if self.opt['mesh_file_type'] == 'h5':
                    with CheckpointFile(os.path.join(self.root, "processed", f"mesh_{n_meshpoints}.h5"), 'r') as mesh_file:
                        setattr(self, f"mesh_{n_meshpoints}",
                                mesh_file.load_mesh(f"ref_mesh_{n_meshpoints}"))

        elif opt['mesh_file_type'] == 'bin':
            self.init_mesh()

        SolverFine = get_solve_firedrake_class(opt)
        self.PDESolver_fine = SolverFine(opt, self.dim, self.fine_mesh)

        for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
            SolverCoarse = get_solve_firedrake_class(opt)
            setattr(self, f"PDESolver_coarse_{n_meshpoints}", SolverCoarse(opt, self.dim, getattr(self, f"mesh_{n_meshpoints}")))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def init_mesh(self):
        # Create PDE solvers
        opt = self.opt

        if opt['mesh_geometry'] == 'rectangle':
            mesh_scale = self.opt.get('mesh_scale', 1.0)

            if self.dim == 1:
                self.fine_mesh = UnitIntervalMesh(self.opt['eval_quad_points'] - 1, name="fine_mesh")
            elif self.dim == 2:
                self.fine_mesh = UnitSquareMesh(self.opt['eval_quad_points'] - 1, self.opt['eval_quad_points'] - 1,
                                                name="fine_mesh")
            elif self.dim == 3:
                self.fine_mesh = UnitCubeMesh(self.opt['eval_quad_points'] - 1,
                                          self.opt['eval_quad_points'] - 1,
                                          self.opt['eval_quad_points'] - 1,
                                          name="fine_mesh")

            # Scale mesh coordinates
            self.fine_mesh.coordinates.dat.data[:] *= mesh_scale
            self.initial_fine_coords = self.fine_mesh.coordinates.copy(deepcopy=True)

            for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
                if self.dim == 1:
                    mesh = UnitIntervalMesh(n_meshpoints - 1, name=f"ref_mesh_{n_meshpoints}")
                elif self.dim == 2:
                    mesh = UnitSquareMesh(n_meshpoints - 1, n_meshpoints - 1, name=f"ref_mesh_{n_meshpoints}")
                elif self.dim == 3:
                    mesh = UnitCubeMesh(n_meshpoints - 1, n_meshpoints - 1, n_meshpoints - 1, name=f"ref_mesh_{n_meshpoints}")

                # Scale mesh coordinates
                mesh.coordinates.dat.data[:] *= mesh_scale
                setattr(self, f"mesh_{n_meshpoints}", mesh)

                # Store initial coordinates
                setattr(self, f"initial_mesh_coords_{n_meshpoints}",
                        getattr(self, f"mesh_{n_meshpoints}").coordinates.copy(deepcopy=True))

    def reset_mesh_coordinates(self):
        if hasattr(self, 'initial_fine_coords'):
            self.fine_mesh.coordinates.assign(self.initial_fine_coords)
        for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
            if hasattr(self, f"initial_mesh_coords_{n_meshpoints}"):
                getattr(self, f"mesh_{n_meshpoints}").coordinates.assign(getattr(self, f"initial_mesh_coords_{n_meshpoints}"))

    def process(self):
        opt = self.opt

        # First initialize the mesh regardless of file type
        self.init_mesh()

        # Save the meshes to HDF5 files if needed
        if self.opt['mesh_file_type'] == 'h5':
            for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
                with CheckpointFile(os.path.join(self.root, "processed", f"mesh_{n_meshpoints}.h5"), 'w') as mesh_file:
                    mesh_file.save_mesh(getattr(self, f"mesh_{n_meshpoints}"))

        SolverFine = get_solve_firedrake_class(opt)
        self.PDESolver_fine = SolverFine(opt, self.dim, self.fine_mesh)

        for n_meshpoints in range(self.min_mesh_points, self.max_mesh_points + 1):
            SolverCoarse = get_solve_firedrake_class(opt)
            setattr(self, f"PDESolver_coarse_{n_meshpoints}", SolverCoarse(opt, self.dim, getattr(self, f"mesh_{n_meshpoints}")))

        self.x_fine_shared = torch.tensor(self.fine_mesh.coordinates.dat.data_ro)
        
        # Map fine mesh to canonical ordering
        if opt['mesh_geometry'] == 'rectangle':
            if self.dim == 1:
                self.mapping_dict_fine, self.mapping_tensor_fine, X_fd_grid_fine, X_fd_vec_fine = map_firedrake_to_cannonical_ordering_1d(self.x_fine_shared, self.opt['eval_quad_points'])
            elif self.dim == 2:
                self.mapping_dict_fine, self.mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine = map_firedrake_to_cannonical_ordering_2d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'])
            elif self.dim == 3:
                self.mapping_dict_fine, self.mapping_tensor_fine, X_fd_grid_fine, Y_fd_grid_fine, Z_fd_grid_fine, X_fd_vec_fine, Y_fd_vec_fine, Z_fd_vec_fine = map_firedrake_to_cannonical_ordering_3d(self.x_fine_shared, self.opt['eval_quad_points'], self.opt['eval_quad_points'], self.opt['eval_quad_points'])
        
            custom_attributes = {
                'mapping_dict_fine': self.mapping_dict_fine,
                'mapping_tensor_fine': self.mapping_tensor_fine,
                'orig_opt': opt.as_dict() if type(opt)!=dict else opt
            }
        else:
            custom_attributes = {
                'orig_opt': opt.as_dict() if type(opt)!=dict else opt
            }

        torch.save(custom_attributes, os.path.join(self.root, "processed", "custom_attributes.pt"))

        data_list = []
        for idx in range(self.num_data):
            # Choose random mesh size
            n = self.opt['mesh_dims_train'][np.random.choice(len(self.opt['mesh_dims_train']))][0] if self.train_test == "train" \
                else self.opt['mesh_dims_test'][np.random.choice(len(self.opt['mesh_dims_test']))][0]
            m = n
            n_meshpoints = n
            coarse_mesh = getattr(self, f"mesh_{n_meshpoints}")

            data = firedrake_mesh_to_PyG(coarse_mesh)

            # Map coarse to canonical ordering
            x_comp = torch.tensor(coarse_mesh.coordinates.dat.data_ro)
            if self.dim == 1:
                data.mapping_dict, data.mapping_tensor, X_fd_grid, X_fd_vec = map_firedrake_to_cannonical_ordering_1d(x_comp, n)
            elif self.dim == 2:
                data.mapping_dict, data.mapping_tensor, X_fd_grid, Y_fd_grid, X_fd_vec, Y_fd_vec = map_firedrake_to_cannonical_ordering_2d(x_comp, n, m)
            elif self.dim == 3:
                data.mapping_dict, data.mapping_tensor, X_fd_grid, Y_fd_grid, Z_fd_grid, X_fd_vec, Y_fd_vec, Z_fd_vec = map_firedrake_to_cannonical_ordering_3d(x_comp, n, m, n)

            pde_solver_coarse = getattr(self, f"PDESolver_coarse_{n_meshpoints}")

            # Get PDE specific parameters
            #set number of gaussians
            mix_num_gauss = np.random.choice(self.opt['num_gauss_range'])
            pde_solver_coarse.mix_num_gauss = mix_num_gauss
            pde_params = pde_solver_coarse.get_pde_params(idx, self.num_data, mix_num_gauss)

            # Pass some global parameters from opt to pde_params for the solvers
            pde_params['mon_power'] = opt['mon_power']
            pde_params['monitor_type'] = opt['monitor_type']
            pde_params['mon_reg'] = opt['mon_reg']
            pde_params['num_gauss'] = mix_num_gauss
            pde_params['mix_num_gauss'] = mix_num_gauss
            pde_params['eval_quad_points'] = opt['eval_quad_points']
            # Store pde_params in the data object
            data.pde_params = pde_params

            #sample data, update solver and solve PDE
            pde_solver_coarse.update_solver(pde_params)
            u = pde_solver_coarse.solve()

            #save inputs to GNN
            data.u_tensor = torch.from_numpy(u.dat.data)
            Hessian_Frob_u = pde_solver_coarse.get_Hessian_Frob_norm()
            data.Hessian_Frob_u_tensor = torch.from_numpy(Hessian_Frob_u.dat.data)

            # For Poisson save the exact f
            if opt['pde_type'] == 'Poisson':
                pde_data = pde_solver_coarse.get_pde_data(pde_params)
                pde_fs = pde_solver_coarse.get_pde_function_space()
                f_data = project(pde_data['f'], pde_fs)
                data.f_tensor = torch.from_numpy(f_data.dat.data)

            self.PDESolver_fine.update_solver(pde_params)
            uu_ref =  self.PDESolver_fine.solve()

            #Deform mesh using MMPDE/MA
            if self.dim == 1:
                data.x_ma, data.ma_its, data.build_time = deform_mesh_mmpde1d(x_comp, self.n, pde_params)
            elif self.dim == 2:
                x_ma, data.ma_its, data.build_time = deform_mesh_ma2d(x_comp, coarse_mesh, pde_solver_coarse, u, Hessian_Frob_u, opt, pde_params, SolverCoarse)
                data.x_ma = torch.from_numpy(x_ma)
            elif self.dim == 3:
                x_ma, data.ma_its, data.build_time = deform_mesh_ma3d(x_comp, coarse_mesh, pde_solver_coarse, u, Hessian_Frob_u, opt, pde_params, SolverCoarse)

            data.x_ma = torch.from_numpy(x_ma)

            # Build suffix for PDE data
            filename_suffix_coarse = f"dim_{self.dim}" \
                              f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{int(mix_num_gauss)}gauss_{idx}_pde_data_coarse.{self.opt['mesh_file_type']}"
            filename_suffix_fine = f"dim_{self.dim}" \
                              f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{int(mix_num_gauss)}gauss_{idx}_pde_data_fine.{self.opt['mesh_file_type']}"

            pde_data_file_coarse = os.path.join(self.root, "processed", filename_suffix_coarse)
            pde_data_file_fine = os.path.join(self.root, "processed", filename_suffix_fine)

            # ----- Save PDE solution + mesh if desired -----
            # HDF5 or BIN
            if self.opt['mesh_file_type'] == 'h5':
                with CheckpointFile(pde_data_file_coarse, 'w') as pde_file:
                    pde_file.save_mesh(data.mesh)
                    pde_file.save_function(u, name="u_coarse")
                with CheckpointFile(pde_data_file_fine, 'w') as pde_file:
                    pde_file.save_mesh(data.mesh_deformed)
                    pde_file.save_function(uu_ref, name="uu_ref")
            elif self.opt['mesh_file_type'] == 'bin':
                save_function(u, pde_data_file_coarse)
                save_function(uu_ref, pde_data_file_fine)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data_list = [data.apply(to_float32) for data in data_list]
        torch.save(data_list, self.processed_paths[0])


    def get(self, idx):
        data = self.data_list[idx]
        num_gauss = data.pde_params['num_gauss']
        
        # Build file suffixes
        filename_suffix_coarse = f"dim_{self.dim}" \
                         f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{int(num_gauss)}gauss_{idx}_pde_data_coarse.{self.opt['mesh_file_type']}"
        filename_suffix_fine = f"dim_{self.dim}" \
                         f"_mon_{data.pde_params['mon_power']}_reg_{data.pde_params['mon_reg']}_{int(num_gauss)}gauss_{idx}_pde_data_fine.{self.opt['mesh_file_type']}"
        
        # Add patch indices if patches are enabled
        if self.use_patches and hasattr(data, 'coarse_mesh') and self.dim == 2:
            node_coords = data.coarse_mesh.coordinates.dat.data_ro
            data.patch_indices = self.get_patch_indices(node_coords)
            
        pde_data_file_coarse = os.path.join(self.root, "processed", filename_suffix_coarse)
        pde_data_file_fine = os.path.join(self.root, "processed", filename_suffix_fine)

        if self.dim == 1:
            n_meshpoints = data.num_nodes
        elif self.dim == 2:
            n_meshpoints = int(np.sqrt(data.num_nodes))
        elif self.dim == 3:
            n_meshpoints = int(np.cbrt(data.num_nodes))

        # Load or create meshes based on file type
        if self.opt['mesh_file_type'] == 'h5':
            with CheckpointFile(os.path.join(self.root, "processed", f"mesh_{n_meshpoints}.h5"), 'r') as mesh_file:
                coarse_mesh = pde_file.load_mesh(f"ref_mesh_{n_meshpoints}")
                u_coarse_reg = pde_file.load_function(self.coarse_mesh, "u_coarse")

            with CheckpointFile(pde_data_file_fine, 'r') as pde_file:
                uu_ref = pde_file.load_function(self.fine_mesh, "uu_ref")

        elif self.opt['mesh_file_type'] == 'bin':
            coarse_mesh = getattr(self, f"mesh_{n_meshpoints}")
            if self.opt['pde_type'] == 'Poisson':
                u_coarse_reg = load_function(coarse_mesh, pde_data_file_coarse, family="CG", degree=1)
                uu_ref = load_function(self.fine_mesh, pde_data_file_fine, family="CG", degree=1)

        data.uu_ref = uu_ref
        data.u_coarse_reg = u_coarse_reg
        data.coarse_mesh = coarse_mesh

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.get(idx)

    def indices(self):
        return list(range(self.__len__()))


def mixed_custom_collate(data_list, collate_keys=[]):
    # collate_keys = ['ma_its', 'u_true_MA_tensor', 'f_fine_tensor', 'successful_eval',
    #                    'uu_fine_tensor', 'eval_errors', 'u_true_fine_tensor',
    #                    'corner_nodes', 'x_comp', 'x_phys',
    # 'f_MA_tensor', 'pde_params', 'uu_tensor', 'edge_index', 'u_true_tensor', 'uu_MA_tensor', 'f_tensor']
    collate_keys = ['u_tensor', 'Hessian_Frob_u_tensor', 'f_tensor', 'uu_ref', 'x_phys', 'x_ma', 'ma_its', 'build_time']#, 'pde_params']

    batch_data = Batch()
    # Initialize containers for all attributes
    concat_attrs = {}  # Attributes that can be concatenated directly
    special_handling_attrs = {}  # Attributes that need special handling

    # Iterate over each data object
    for data in data_list:
        for key in data.keys():
            # Assuming `concat_attrs` is a dictionary of lists
            if key in collate_keys:  # Attributes that can be concatenated
                if key not in concat_attrs:
                    concat_attrs[key] = []
                concat_attrs[key].append(data[key])
            else:  # Special handling attributes
                if key not in special_handling_attrs:
                    special_handling_attrs[key] = []
                special_handling_attrs[key].append(data[key])

    # Concatenate concatenable attributes
    for key, values in concat_attrs.items():
            batch_data[key] = torch.cat(values, dim=0)

    # Handle non-concatenable attributes
    for key, values in special_handling_attrs.items():
        batch_data[key] = values

    # Create batch indices for the concatenated data
    batch_data.batch = torch.tensor([i for i, data in enumerate(data_list) for _ in range(data.num_nodes)])

    return batch_data


def analyze_data_keys(data_list):
    all_keys = set()
    key_presence = {}

    # Gather all keys and initialize tracking for their presence across data items
    for data in data_list:
        for key in data.keys():
            all_keys.add(key)
            if key not in key_presence:
                key_presence[key] = [0] * len(data_list)

    # Mark presence of each key in each data item
    for i, data in enumerate(data_list):
        for key in data.keys():
            key_presence[key][i] = 1

    # Report results
    consistent_keys = {key for key, presences in key_presence.items() if all(presences)}
    inconsistent_keys = {key for key, presences in key_presence.items() if not all(presences)}

    print("Consistent keys across all data items:", consistent_keys)
    print("Inconsistent keys (missing in some data items):", inconsistent_keys)

    return consistent_keys, inconsistent_keys


if __name__ == "__main__":
    opt = get_params()
    opt = run_params(opt)
    rand_seed = np.random.randint(3, 10000)
    opt['seed'] = rand_seed
    opt['pde_type'] = 'Poisson'

    opt['mesh_geometry'] = 'rectangle'
    opt['mesh_file_type'] = 'bin'
    opt['data_type'] = 'randg_mix'
    dim = 3
    d3_dim = 10
    opt['mesh_dims'] = [d3_dim, d3_dim, d3_dim]
    opt['mon_reg'] = 0.1
    opt['rand_gauss'] = True
    opt['num_train'] = 10
    opt['num_test'] = 10
    opt['anis_gauss'] = False
    opt['monitor_type'] = 'monitor_hessian'
    if opt['anis_gauss']:
        opt['mesh_dims_train'] = [[15, 15], [20, 20]]
        opt['mesh_dims_test'] = [[i, i] for i in range(12, 24, 1)]
        opt['num_gauss_range'] = [1, 2, 3, 5, 6]
    else:
        opt['mesh_dims_train'] = [[d3_dim, d3_dim, d3_dim]]
        opt['mesh_dims_test'] = [[d3_dim, d3_dim, d3_dim]]
        opt['num_gauss_range'] = [2, 3]

    opt['eval_quad_points'] = 30
    opt['monitor_alpha'] = 20.0

    for train_test in ['train', 'test']:
        opt = make_data_name(opt, train_test)
        if train_test == 'train':
            dataset = MeshInMemoryDataset_Mixed(f"../data/{opt['data_name']}", opt['num_train'], opt['mesh_dims'], train_test, opt)
        elif train_test == 'test':
            dataset = MeshInMemoryDataset_Mixed(f"../data/{opt['data_name']}", opt['num_test'], opt['mesh_dims'], train_test, opt)

        consistent_keys, inconsistent_keys = analyze_data_keys(dataset.data_list)

        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False,
                                  exclude_keys=exclude_keys, follow_batch=follow_batch, generator=torch.Generator(device=opt['device']))

        for i, data in enumerate(loader):
            print(f"Batch {i} retrieved successfully.")
            print(data)

        if dim == 1:
            pass
            plot_initial_dataset_1d(dataset, opt)
        elif dim == 2:
            plot_initial_dataset_2d(dataset, opt)
        elif dim == 3:
            plot_initial_dataset_3d(dataset, opt)