import os
import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import pyvista as pv

from firedrake import VectorFunctionSpace, Function, dx, inner, assemble, sqrt, UnitIntervalMesh, UnitSquareMesh
from firedrake.pyplot import plot, tripcolor, triplot
from firedrake import *

from src.utils_main import vizualise_grid_with_edges
from src.data_mixed_loader import Mixed_DataLoader
from src.utils_eval import update_mesh_coords
from firedrake_difFEM.solve_poisson import poisson2d_fmultigauss_bcs, poisson1d_fmultigauss_bcs, plot_solutions

def plot_trained_dataset_2d(dataset, model, opt, show_mesh_evol_plots=False):
    # TODO: This is only supported for Poisson currently
    if opt['pde_type'] != 'Poisson':
        NotImplementedError("Plotting for non-Poisson not supported yet")

    # Create a DataLoader with batch size of 1 to load one data point at a time
    # loader = DataLoader(dataset, batch_size=1)
    if opt['data_type'] == 'randg_mix':# and opt['batch_size'] > 1:
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False, exclude_keys=exclude_keys, follow_batch=follow_batch)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    #figure for FEM on regular mesh
    fig0, axs0 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs0 = axs0.ravel()
    fig0.suptitle('FEM on regular mesh', fontsize=20)
    fig0.tight_layout()

    # figure for MA mesh
    fig1, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs1 = axs.ravel()
    fig1.suptitle('MA mesh', fontsize=20)
    fig1.tight_layout()

    # #figure for FEM on MA mesh
    fig2, axs2 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs2 = axs2.ravel()
    fig2.suptitle('FEM on MA mesh', fontsize=20)
    fig2.tight_layout()

    # figure for MLmodel mesh
    fig3, axs3 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs3 = axs3.ravel()
    fig3.suptitle('MLmodel mesh', fontsize=20)
    fig3.tight_layout()

    # figure for FEM on MLmodel mesh
    fig4, axs4 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    axs4 = axs4.ravel()
    fig4.suptitle('FEM on MLmodel mesh', fontsize=20)
    fig4.tight_layout()

    # # figure for error of MA mesh versus regular mesh
    # fig5, axs5 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    # axs5 = axs5.ravel()
    # fig5.suptitle('Error of MA mesh versus regular mesh - CURRENTLY NOT WELL DEFINED', fontsize=20)
    # fig5.tight_layout()
    #
    # # figure for error of MLmodel mesh versus regular mesh
    # fig6, axs6 = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))  # adjust as necessary
    # axs6 = axs6.ravel()
    # fig6.suptitle('Error of MLmodel mesh versus regular mesh - CURRENTLY NOT WELL DEFINED', fontsize=20)
    # fig6.tight_layout()

    # Loop over the dataset
    for i, data in enumerate(loader):
        data.idx = i
        if i == 25:
            break
    # idx_list = [i for i in range(25, 50, 1)]
    # for j, data in enumerate(loader):
    #     data.idx = j
    #     if j not in idx_list:
    #         continue
    #     else:
    #         i = j

        if opt['test_idxs']:
            if i not in opt['test_idxs']:
                continue  # skip to next batch
            else:
                print(f"Running on batch {i} of {opt['test_idxs']}")

        # todo check this annoying property of PyG I believe, making indexing necessary
        if opt['data_type'] == 'randg_mix':
            mesh = data.coarse_mesh[0]
            c_list = data.batch_dict[0]['pde_params']['centers']
            s_list = data.batch_dict[0]['pde_params']['scales']
        elif opt['data_type'] == 'randg':
            mesh = dataset.coarse_mesh
            c_list = data.pde_params['centers'][0]
            s_list = data.pde_params['scales'][0]
        else:
            mesh = dataset.coarse_mesh
            c_list = data.pde_params['centers'][0]
            s_list = data.pde_params['scales'][0]

        # 0) plot the FEM on regular mesh
        colors = tripcolor(data.uu_ref[0], axes=axs0[i])  # , shading='gouraud', cmap='viridis')

        # Convert PyG graph to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # 1) plot the MA mesh
        # Get node positions from the coordinates attribute in the PyG graph
        x = data.x_ma
        positions = {i: x[i].tolist() for i in range(x.shape[0])}
        nx.draw(G, pos=positions, ax=axs1[i], node_size=1, width=0.5, with_labels=False)

        # #2) plot the FEM on MA (target phys) mesh
        update_mesh_coords(mesh, x)
        uu_ma, u_true_ma, _ = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list, rand_gaussians=False)
        colors = tripcolor(uu_ma, axes=axs2[i])#, shading='gouraud', cmap='viridis')

        #3) Get the model deformed mesh solution
        update_mesh_coords(mesh, data.x_comp) # Reset the mesh coordinates
        if opt['model'] in ['backFEM_2D', 'GNN', 'UM2N_T_GRAND'] and show_mesh_evol_plots:
            model.plot_evol_flag = True

        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'modular':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss_torch':
            # raise NotImplementedError("PDE loss evaluation, please switch to pde_loss_firedrake.")
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss_firedrake':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'UM2N_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'mixed_UM2N_pde_loss_firedrake':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss_regularised':
            MLmodel_coords = model(data).to('cpu').detach().numpy()

        if opt['model'] in ['backFEM_2D', 'GNN'] and show_mesh_evol_plots:
            model.plot_evol_flag = False

        # 3) plot the MLmodel mesh and evol if applicable
        positions = {i: MLmodel_coords[i].tolist() for i in range(MLmodel_coords.shape[0])}
        nx.draw(G, pos=positions, ax=axs3[i], node_size=1, width=0.5, with_labels=False)

        # 4) plot the FEM on MLmodel mesh
        update_mesh_coords(mesh, MLmodel_coords)
        # solve poisson
        uu_gnn, u_true_gnn, f = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                      rand_gaussians=False)
        colors = tripcolor(uu_gnn, axes=axs4[i])#, shading='gouraud', cmap='viridis')

        # 5) plot the error of MA mesh versus regular mesh
        # error = uu_ma - data.uu[0]
        # colors = tripcolor(error, axes=axs5[i])#, shading='gouraud', cmap='viridis')
        # error = Function(uu_ma.function_space())
        # error.assign(uu_ma - data.uu[0])
        # tripcolor(error, axes=axs5[i])

        # 6) plot the error of MLmodel mesh versus regular mesh
        # error = uu_gnn - data.uu[0]
        # colors = tripcolor(error, axes=axs6[i])#, shading='gouraud', cmap='viridis')
        # error = Function(uu_gnn.function_space())
        # error.assign(uu_gnn - data.uu[0])
        # tripcolor(error, axes=axs6[i])

    if opt['wandb_log_plots']:
        wandb.log({'fem_on_x_comp': wandb.Image(fig0)})
        wandb.log({'MA mesh': wandb.Image(fig1)})
        wandb.log({'fem_on_MA': wandb.Image(fig2)})
        wandb.log({'MLmodel mesh': wandb.Image(fig3)})
        wandb.log({'fem_on_MLmodel': wandb.Image(fig4)})

    if show_mesh_evol_plots:
        if opt['model'] in ['GNN']:
            if opt['wandb_log_plots']:
                wandb.log({'mesh_evol': wandb.Image(model.mesh_fig)})
                for idx in opt['evol_plot_idxs']:
                    # wandb.log({f'mesh_evol_{idx}' : wandb.Image(model.mesh_fig_dict[idx])})
                    wandb.log({f'attention_weights_{idx}': wandb.Image(model.att_fig_dict[idx][0])})

    if opt['show_dataset_plots']:
        plt.show()


def plot_individual_meshes(dataset, model, opt):
    dim = len(opt['mesh_dims'])

    # update mesh coordinates
    # visualise first N meshes and results
    N = 1
    loader = DataLoader(dataset, batch_size=1)
    for i, data in enumerate(loader):
        print(f"visualising mesh {i}")
        if opt['show_plots']:
            vizualise_grid_with_edges(data.x_ma, data.edge_index, opt, boundary_nodes=data.boundary_nodes)
            vizualise_grid_with_edges(data.x_comp, data.edge_index, opt, boundary_nodes=data.boundary_nodes)
            learned_mesh = model(data)
            if opt['fix_boundary']:
                mask = ~data.to_boundary_edge_mask * ~data.to_corner_nodes_mask * ~data.diff_boundary_edges_mask
                edge_index = data.edge_index[:, mask]
                # need to add self loops for the corner nodes or they go to zero
                corner_nodes = torch.cat([torch.from_numpy(arr) for arr in data.corner_nodes]).repeat(2, 1)
                edge_index = torch.cat([edge_index, corner_nodes], dim=1)

            _ = vizualise_grid_with_edges(learned_mesh, edge_index, opt,
                                                     boundary_nodes=data.boundary_nodes, node_labels=False,
                                                     node_boundary_map=data.node_boundary_map, corner_nodes=data.corner_nodes, edge_weights=model.gnn_deformer.convs[-1].alphas)

        # update firedrake computational mesh with deformed coordinates
        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach().numpy()
        elif opt['loss_type'] == 'pde_loss':
            coeffs, MLmodel_coords, sol = model(data)
            MLmodel_coords = MLmodel_coords.to('cpu').detach().numpy()

        mesh = dataset.mesh
        update_mesh_coords(mesh, MLmodel_coords)
        # solve poisson
        c_list = data.pde_params['centers'][0] #todo check this annoying property of PyG I believe, making indexing necessary
        s_list = data.pde_params['scales'][0]

        if dim == 1:
            uu, u_true, f = poisson1d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                  rand_gaussians=False)

        elif dim == 2:
            uu, u_true, f = poisson2d_fmultigauss_bcs(mesh, c_list=c_list, s_list=s_list,
                                                                  rand_gaussians=False)
            plot_solutions(uu, u_true)

        if i == N-1:
            break

"""3D visualization and evaluation utilities."""

def plot_trained_dataset_3d(dataset, model, opt, show_mesh_evol_plots=False):
    """Plot 3D mesh results including solution fields and mesh quality metrics.

    Args:
        dataset: Dataset containing the mesh data
        model: Trained model to evaluate
        opt: Options dictionary
        show_mesh_evol_plots: Whether to show mesh evolution plots
    """
    if opt['data_type'] == 'randg_mix':  # and opt['batch_size'] > 1:
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False, exclude_keys=exclude_keys,
                                  follow_batch=follow_batch)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):

        # Get model prediction
        with torch.no_grad():
            pred = model(data)

        # Plot original coarse mesh
        coarse_mesh = data.coarse_mesh
        vtk_orig = VTKFile(f"mesh_{i}_original.pvd")
        vtk_orig.write(data.u_coarse_reg[0], name="solution")

        # Save deformed mesh to VTK
        vtk_deformed = VTKFile(f"mesh_{i}_deformed.pvd")
        coarse_mesh[0].coordinates.dat.data[:] = data.x_ma.detach().cpu().numpy()
        vtk_deformed.write(data.uu_ref[0], name="solution")

        # Get mesh data for PyVista
        orig_coords = coarse_mesh[0].coordinates.dat.data_ro.copy()
        orig_cells = coarse_mesh[0].coordinates.cell_node_map().values

        # Create PyVista grid for original mesh
        cells = np.column_stack((np.full((len(orig_cells), 1), 4), orig_cells)).astype(
            np.int32)  # Add cell type (4 for tetra)
        cell_types = np.full(len(orig_cells), 10, dtype=np.int32)  # VTK_TETRA = 10
        orig_grid = pv.UnstructuredGrid(cells, cell_types, orig_coords.astype(np.float64))
        # orig_grid.point_data['Solution'] = data.u_coarse_reg[0].dat.data_ro.copy()
        V_orig = FunctionSpace(coarse_mesh[0], "CG", 1)
        u_orig = Function(V_orig)
        u_orig.interpolate(data.u_coarse_reg[0])
        orig_grid.point_data['Solution'] = u_orig.dat.data_ro.copy()

        # Plot deformed mesh
        deformed_coords = data.x_ma.detach().cpu().numpy().astype(np.float64)
        deformed_grid = pv.UnstructuredGrid(cells, cell_types, deformed_coords)  # Use same cells

        # Update mesh coordinates for solution computation
        coarse_mesh[0].coordinates.dat.data[:] = deformed_coords

        # Create function space on coarse mesh and interpolate solution
        V_coarse = FunctionSpace(coarse_mesh[0], "CG", 1)
        u_coarse = Function(V_coarse)
        u_coarse.interpolate(data.uu_ref[0])
        deformed_grid.point_data['Solution'] = u_coarse.dat.data_ro.copy()

        def setup_plotter(p):
            # Plot original mesh
            p.subplot(0, 0)
            p.add_mesh(orig_grid, scalars="Solution", show_edges=True)
            p.add_text("Original Mesh")

            # Plot deformed mesh
            p.subplot(0, 1)
            p.add_mesh(deformed_grid, scalars="Solution", show_edges=True)
            p.add_text("Deformed Mesh")

            # Link the camera views
            p.link_views()

        # Show interactive plot if requested
        if opt['show_dataset_plots']:
            plotter = pv.Plotter(shape=(1, 2))
            setup_plotter(plotter)
            plotter.show()

        # Save to wandb if enabled
        if opt['wandb'] and opt['wandb_log_plots']:
            plotter = pv.Plotter(shape=(1, 2), off_screen=True)
            setup_plotter(plotter)
            plotter.screenshot(f'mesh_{i}.png')
            wandb.log({f"mesh_{i}": wandb.Image(f'mesh_{i}.png')})
            os.remove(f'mesh_{i}.png')


def compute_3d_mesh_quality(vertices, faces):
    """Compute quality metrics for 3D tetrahedral mesh.

    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx4 array of tetrahedral indices

    Returns:
        metrics: Dictionary of mesh quality metrics
    """
    # Get tetrahedra vertices
    tets = vertices[faces]

    # Compute volumes
    v1 = tets[:, 1] - tets[:, 0]
    v2 = tets[:, 2] - tets[:, 0]
    v3 = tets[:, 3] - tets[:, 0]
    volumes = np.abs(np.sum(v1 * np.cross(v2, v3), axis=1)) / 6.0

    # Compute edge lengths
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edge = tets[:, j] - tets[:, i]
            edges.append(np.sqrt(np.sum(edge ** 2, axis=1)))
    edges = np.array(edges).T

    # Compute aspect ratio
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    aspect_ratio = max_edge / min_edge

    # Compute dihedral angles
    def compute_dihedral_angle(v1, v2, v3, v4):
        n1 = np.cross(v2 - v1, v3 - v1)
        n2 = np.cross(v2 - v4, v3 - v4)
        n1 = n1 / np.linalg.norm(n1, axis=1)[:, np.newaxis]
        n2 = n2 / np.linalg.norm(n2, axis=1)[:, np.newaxis]
        cos_angle = np.sum(n1 * n2, axis=1)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Compute all dihedral angles
    dihedrals = []
    for i in range(4):
        for j in range(i + 1, 4):
            remaining = [k for k in range(4) if k != i and k != j]
            angle = compute_dihedral_angle(tets[:, i], tets[:, j],
                                           tets[:, remaining[0]], tets[:, remaining[1]])
            dihedrals.append(angle)
    dihedrals = np.array(dihedrals).T

    return {
        'volumes': volumes,
        'aspect_ratio': aspect_ratio,
        'min_volume': np.min(volumes),
        'max_aspect_ratio': np.max(aspect_ratio),
        'min_dihedral': np.min(dihedrals),
        'max_dihedral': np.max(dihedrals)
    }