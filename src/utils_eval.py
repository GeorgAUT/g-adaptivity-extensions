import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
import time
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib
#matplotlib.use('TkAgg')  # Switch to a different backend to avoid PyCharm's custom backend issues
import matplotlib.pyplot as plt
import pyvista as pv

from firedrake import VectorFunctionSpace, Function, dx, inner, assemble, sqrt, UnitIntervalMesh, UnitSquareMesh
from firedrake.pyplot import plot, tripcolor, triplot
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.exceptions import ConvergenceError

from firedrake_difFEM.solve_poisson import poisson2d_fgauss_b0, poisson2d_fmultigauss_bcs, poisson1d_fmultigauss_bcs, plot_solutions
from utils_main import vizualise_grid_with_edges, inner_progress

from data_mixed_loader import Mixed_DataLoader
from pde_solvers import get_solve_firedrake_class


def eval_firedrake_fct(uu, u_true, p=2):
    # Compute and return L1 / L2 error
    if p==1:
        Lp = assemble(sqrt(inner(uu - u_true, uu - u_true)) * dx)
    elif p==2:
        Lp = assemble(inner(uu - u_true, uu - u_true) * dx)
    return Lp



def evaluate_error_np(uu, u_true, x):
    # Calculate the lengths of each interval
    dx = np.diff(x)

    # Calculate local errors
    local_L2_errors = ((uu - u_true) ** 2)[1:] + ((uu - u_true) ** 2)[:-1]
    local_L1_errors = np.abs(uu - u_true)[1:] + np.abs(uu - u_true)[:-1]

    # Apply the trapezium rule to calculate global error
    L2_error = np.sqrt(np.sum(local_L2_errors * dx) / 2)
    L1_error = np.sum(local_L1_errors * dx) / 2

    return L1_error, L2_error

def evaluate_error_np_2d(uu, u_true, x):
    #x=np.transpose(x1)
    # Calculate the lengths of each interval in both dimensions
    dx = np.diff(x[0], axis=1)[:-1,:]
    dy = np.diff(x[1], axis=0)[:,:-1]
    uu = uu.reshape(x[0].shape)
    u_true = u_true.reshape(x[0].shape)
    error=uu-u_true

    # Calculate local errors
    local_L2_errors = (error ** 2)[:-1, 1:] + (error ** 2)[1:, :-1]
    local_L2_errors += (error ** 2)[1:, 1:] + (error ** 2)[:-1, :-1]
    local_L1_errors = np.abs(error)[:-1, 1:] + np.abs(error)[1:, :-1]
    local_L1_errors += np.abs(error)[1:, 1:] + np.abs(error)[:-1, :-1]

    # Apply the trapezium rule to calculate global error
    L2_error = np.sqrt(np.sum(local_L2_errors * dx * dy) / 4)
    L1_error = np.sum(local_L1_errors * dx * dy) / 4

    return L1_error, L2_error


def calculate_error_reduction(e_initial, e_adapted):
    ''' Calculate the percentage of error reduction '''
    if e_adapted == 0.:
        return None
    else:
        return (e_adapted - e_initial) / e_initial * 100


def update_mesh_coords(mesh, new_coords):
    V = VectorFunctionSpace(mesh, 'P', 1)
    new_coordinates = Function(V)
    dim = len(new_coords.shape)
    if dim == 1 or new_coords.shape[1] == 1:
        new_coords = new_coords.squeeze()
        new_coordinates.dat.data[:] = new_coords
    elif dim == 2 or dim == 3:
        new_coordinates.dat.data[:] = new_coords
    mesh.coordinates.assign(new_coordinates)


def firedrake_call_fct_handler(fd_fct, point_list):
    '''to handle this unresolved firedrake issue:
    https://github.com/firedrakeproject/firedrake/issues/2359
    ie firedrake.function.PointNotInDomainError: domain <Mesh #2> does not contain point [0.6]
    nb other option is changing tolerance:
                #uu_MA_coarse.function_space().mesh().coordinates.dat.data_ro
            #{PointNotInDomainError}domain <Mesh #2> does not contain point [0.34 0.  ]
            # uu_MA_coarse.at(np.array([0.34,0.]), tolerance=0.1)
    '''
    fd_fct_eval = np.array(fd_fct.at(point_list, dont_raise=True))
    # extract indices where non
    non_idxs = np.where(fd_fct_eval == None)
    print(f"Number of bad eval mesh points: {len(non_idxs)} ie {len(non_idxs) / len(fd_fct_eval)}")
    fd_fct_eval[non_idxs] = 0.

    return fd_fct_eval, non_idxs


def inner_progress(curr, total, width=10, bars=u'▉▊▋▌▍▎▏ '[::-1],
               full='█', empty=' '):
    """Create a progress bar string for inner loops to use in tqdm postfix.
    
    Args:
        curr: Current step
        total: Total steps
        width: Width of the progress bar
        bars: Characters to use for fractional progress
        full: Character for completed sections
        empty: Character for empty sections
    
    Returns:
        Formatted progress bar string
    """
    p = curr / total 
    nfull = int(p * width)
    return "{:>3.0%} |{}{}{}| {:>2}/{}".format(
        p, 
        full * nfull,
        bars[int(len(bars) * ((p * width) % 1))] if nfull < width else '',
        empty * (width - nfull - 1),
        curr, total
    )

def solve_with_retry(pde_solver, max_attempts=3, verbose=False, progress_callback=None):
    """Attempt to solve with multiple retry strategies if initial solve fails.
    
    Args:
        pde_solver: The PDE solver instance
        max_attempts: Maximum number of solution attempts
        verbose: If True, print detailed solver information
    """
    # Only print solver configuration in verbose mode
    if verbose:
        try:
            print(f"\nInitial solver configuration:")
            print(f"Solver type: {type(pde_solver)}")
            if hasattr(pde_solver, 'solver'):
                snes = pde_solver.solver.snes
                print(f"SNES type: {snes.getType()}")
                try:
                    print(f"Line search type: {snes.getLineSearch().getType()}")
                except:
                    pass  # Older PETSc versions might not have this
                try:
                    rtol, atol, stol, maxit = snes.getTolerances()
                    print(f"SNES tolerances: rtol={rtol}, atol={atol}, max_it={maxit}")
                except:
                    pass
        except Exception as e:
            print(f"Could not print solver configuration: {e}")
    
    attempts = 0
    while attempts < max_attempts:
        try:
            result = pde_solver.solve()
            # Print solution stats only in verbose mode
            if verbose:
                try:
                    print(f"Solution min/max: {result.dat.data.min():.3e}/{result.dat.data.max():.3e}")
                except:
                    pass
            return result
        except ConvergenceError as e:
            attempts += 1
            if verbose:
                print(f"\nSolver failure details:")
                print(f"Error message: {str(e)}")
                try:
                    # Try to get more detailed convergence info
                    if hasattr(pde_solver, 'solver'):
                        snes = pde_solver.solver.snes
                        print(f"SNES reason: {snes.getConvergedReason()}")
                        print(f"Number of iterations: {snes.getIterationNumber()}")
                        print(f"Function norm: {snes.getFunctionNorm()}")
                        print(f"Current tolerances: {snes.getTolerances()}")
                except Exception as debug_e:
                    print(f"Could not get detailed convergence info: {debug_e}")
            
            if attempts == max_attempts:
                if verbose:
                    print(f"Failed to converge after {max_attempts} attempts")
                raise e
            
            # Modify solver parameters for next attempt
            if "DIVERGED_FNORM_NAN" in str(e):
                if attempts == 1:
                    if verbose:
                        print("First retry: Using basic line search...")
                    try:
                        snes = pde_solver.solver.snes
                        snes.setLineSearchType('basic')
                        # Try more conservative tolerances
                        rtol, atol, stol, maxit = snes.getTolerances()
                        snes.setTolerances(rtol=rtol*10, atol=atol*10, max_it=maxit*2)
                        snes.setFromOptions()
                    except AttributeError:
                        if verbose:
                            print("Warning: Could not modify SNES parameters")
                elif attempts == 2:
                    if verbose:
                        print("Second retry: Using more robust solver configuration...")
                    try:
                        snes = pde_solver.solver.snes
                        snes.setType('newtontr')  # Try trust region instead
                        snes.setLineSearchType('l2')
                        # Even more conservative tolerances
                        rtol, atol, stol, maxit = snes.getTolerances()
                        snes.setTolerances(rtol=rtol*100, atol=atol*100, max_it=maxit*4)
                        snes.setFromOptions()
                    except AttributeError:
                        if verbose:
                            print("Warning: Could not modify SNES parameters")
            
            if verbose:
                print(f"Solve failed, attempt {attempts}/{max_attempts}. Retrying with modified parameters...")

    return result

def evaluate_model(model, dataset, opt):
    dim = len(dataset.opt['mesh_dims'])

    SolverCoarse = get_solve_firedrake_class(opt)

    if opt['data_type'] == 'randg_mix':
        exclude_keys = ['boundary_nodes_dict', 'mapping_dict', 'node_boundary_map', 'eval_errors', 'pde_params']
        follow_batch = []
        loader = Mixed_DataLoader(dataset, batch_size=1, shuffle=False, exclude_keys=exclude_keys, follow_batch=follow_batch)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results, times, metrics = [], [], []
    successful_evals = 0

    eval_bar = tqdm(enumerate(loader), desc="Evaluating", total=len(loader), position=0, leave=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, data in eval_bar:

        if 'test_idxs' in opt and opt['test_idxs']:
            idx = dataset.data_idx_dict[i] if opt['data_type'] == "RBF" else i
            if idx not in opt['test_idxs']:
                continue

        if opt['data_type'] == 'randg_mix':
            mesh = data.coarse_mesh[0]
            pde_params = data.batch_dict[0]['pde_params']
        else:
            mesh = dataset.coarse_mesh
            pde_params = data.pde_params

        # to catch varying size data
        if dim == 1:
            num_meshpoints = dataset.opt['mesh_dims'][0]
        elif dim == 2:
            num_meshpoints = dataset.opt['mesh_dims'][0]
        elif dim == 3:
            num_meshpoints = int(np.sqrt(data.x_ma.shape[0]))

        data.idx = i

        # Reset mesh coordinates at start of each evaluation
        if opt['mesh_file_type'] == 'bin':
            dataset.reset_mesh_coordinates()

        #1) GRID
        update_mesh_coords(mesh, data.x_comp)

        PDESolver_coarse = SolverCoarse(opt, dim, mesh)
        PDESolver_coarse.update_solver(pde_params)
        try:
            # Define a callback to update the progress bar
            def solve_progress(step, total):
                eval_bar.set_postfix_str(f"Solve: {inner_progress(step, total)}")
                
            if opt['pde_type'] in ['Poisson', 'Burgers']:
                u_grid = solve_with_retry(PDESolver_coarse, verbose=False, progress_callback=solve_progress)
            elif opt['pde_type'] == 'NavierStokes':
                u_grid, p_grid = solve_with_retry(PDESolver_coarse, verbose=False, progress_callback=solve_progress)
            successful_evals += 1
        except Exception as e:
            print(f"Solve failed completely for idx {i}, skipping...")
            print(f"Error: {str(e)}")
            # Reset mesh coordinates after failed solve
            if opt['mesh_file_type'] == 'bin':
                dataset.reset_mesh_coordinates()
            continue

        # TODO: Make this NavierStokes compatible
        if opt['pde_type'] in ['Poisson', 'Burgers']:
            V_HO = PDESolver_coarse.get_pde_function_space(opt['HO_degree'])
        elif opt['pde_type'] == 'NavierStokes':
            V_HO, Q_HO = PDESolver_coarse.get_pde_function_space(opt['HO_degree'])
        # Project reference solution onto HO space
        u_grid_ref = assemble(interpolate(data.uu_ref[0], V_HO))

        if opt['pde_type'] == 'NavierStokes':
            p_grid_ref = assemble(interpolate(data.pp_ref[0], Q_HO))

        L1_grid = eval_firedrake_fct(u_grid, u_grid_ref, p=1)
        L2_grid = np.sqrt(eval_firedrake_fct(u_grid, u_grid_ref, p=2))

        if opt['pde_type'] == 'NavierStokes':
            L1_grid = L1_grid + eval_firedrake_fct(p_grid, p_grid_ref, p=1)
            L2_grid = np.sqrt(L2_grid**2 + eval_firedrake_fct(p_grid, p_grid_ref, p=2))

        #2) MA: Update mesh coordinates and repeat process
        update_mesh_coords(mesh, data.x_ma)
        if opt['pde_type'] in ['Poisson', 'Burgers']:
            PDESolver_coarse = SolverCoarse(opt, dim, mesh)
            PDESolver_coarse.update_solver(pde_params)
            try:
                u_MA = solve_with_retry(PDESolver_coarse, verbose=False)
                successful_evals += 1
            except Exception as e:
                print(f"Solve failed completely for idx {i}, skipping...")
                print(f"Error: {str(e)}")
                # Reset mesh coordinates after failed solve
                if opt['mesh_file_type'] == 'bin':
                    dataset.reset_mesh_coordinates()
                continue

            V_HO = PDESolver_coarse.get_pde_function_space(opt['HO_degree'])
            try:
                # Project reference solution onto HO space
                u_MA_ref = assemble(interpolate(data.uu_ref[0], V_HO))
                # Compute L1 and L2 errors
                L1_MA = eval_firedrake_fct(u_MA, u_MA_ref, p=1)
                L2_MA = np.sqrt(eval_firedrake_fct(u_MA, u_MA_ref, p=2))
            except:
                L1_MA=L1_grid
                L2_MA=L2_grid

            MA_time = data.build_time.item()
            L1_reduction_MA = calculate_error_reduction(L1_grid, L1_MA)
            L2_reduction_MA = calculate_error_reduction(L2_grid, L2_MA)

        elif opt['pde_type'] == 'NavierStokes':
            L1_MA = 0.0
            L2_MA = 0.0
            MA_time = 0.0
            L1_reduction_MA = 0.0
            L2_reduction_MA = 0.0


        #3) Get the model deformed mesh from trained model
        update_mesh_coords(mesh, data.x_comp)
        # start_MLmodel = time.time()
        if opt['loss_type'] == 'mesh_loss':
            MLmodel_coords = model(data).to('cpu').detach()
        elif opt['loss_type'] == 'pde_loss_firedrake':
            MLmodel_coords = model(data).to('cpu').detach()
        elif opt['loss_type'] == 'modular':
            MLmodel_coords = model(data).to('cpu').detach()
        elif opt['loss_type'] == 'UM2N_loss':
            MLmodel_coords = model(data).to('cpu').detach()
        elif opt['loss_type'] == 'mixed_UM2N_pde_loss_firedrake':
            MLmodel_coords = model(data).to('cpu').detach()
        elif opt['loss_type'] == 'pde_loss_regularised':
            MLmodel_coords = model(data).to('cpu').detach()

        MLmodel_time = model.end_MLmodel - model.start_MLmodel

        MLmodel_coords = MLmodel_coords.squeeze() if dim == 1 else MLmodel_coords

        # Update mesh coordinates and repeat process
        update_mesh_coords(mesh, MLmodel_coords)

        PDESolver_coarse = SolverCoarse(opt, dim, mesh)
        PDESolver_coarse.update_solver(pde_params)
        try:
            if opt['pde_type'] in ['Poisson', 'Burgers']:
                u_MLmodel = solve_with_retry(PDESolver_coarse, verbose=False)

            elif opt['pde_type'] == 'NavierStokes':
                u_MLmodel, p_MLmodel = solve_with_retry(PDESolver_coarse, verbose=False)
            successful_evals += 1
        except Exception as e:
            print(f"Solve failed completely for idx {i}, skipping...")
            print(f"Error: {str(e)}")
            # Reset mesh coordinates after failed solve
            if opt['mesh_file_type'] == 'bin':
                dataset.reset_mesh_coordinates()
            continue

        if opt['pde_type'] in ['Poisson', 'Burgers']:
            V_HO = PDESolver_coarse.get_pde_function_space(opt['HO_degree'])
        elif opt['pde_type'] == 'NavierStokes':
            V_HO, Q_HO = PDESolver_coarse.get_pde_function_space(opt['HO_degree'])

        try:
            # Project reference solution onto HO space
            u_MLmodel_ref = assemble(interpolate(data.uu_ref[0], V_HO))

            if opt['pde_type'] == 'NavierStokes':
                p_MLmodel_ref = assemble(interpolate(data.pp_ref[0], Q_HO))

            L1_MLmodel = eval_firedrake_fct(u_MLmodel, u_MLmodel_ref, p=1)
            L2_MLmodel = np.sqrt(eval_firedrake_fct(u_MLmodel, u_MLmodel_ref, p=2))

            if opt['pde_type'] == 'NavierStokes':
                L1_MLmodel = L1_MLmodel + eval_firedrake_fct(p_MLmodel, p_MLmodel_ref, p=1)
                L2_MLmodel = np.sqrt(L2_MLmodel**2 + eval_firedrake_fct(p_MLmodel, p_MLmodel_ref, p=2))
        except:
            L1_MLmodel = L1_grid
            L2_MLmodel = L2_grid

        # Calculate error reduction ratios
        L1_reduction_MLmodel = calculate_error_reduction(L1_grid, L1_MLmodel)
        L2_reduction_MLmodel = calculate_error_reduction(L2_grid, L2_MLmodel)

        # Calculate mesh quality metrics for all meshes
        quality_metrics = {
            **calculate_mesh_quality_metrics(mesh, data.x_comp, 'grid'),
            **calculate_mesh_quality_metrics(mesh, data.x_ma, 'MA'),
            **calculate_mesh_quality_metrics(mesh, MLmodel_coords, 'MLmodel')
        }

        results.append({
            'L1_grid': L1_grid,
            'L2_grid': L2_grid,
            'L1_MA': L1_MA,
            'L2_MA': L2_MA,
            'L1_MLmodel': L1_MLmodel,
            'L2_MLmodel': L2_MLmodel,
            'L1_reduction_MA': L1_reduction_MA,
            'L2_reduction_MA': L2_reduction_MA,
            'L1_reduction_MLmodel': L1_reduction_MLmodel,
            'L2_reduction_MLmodel': L2_reduction_MLmodel
        })

        times.append({
            'MA_time': MA_time,
            'MLmodel_time': MLmodel_time})
        
        metrics.append(quality_metrics)

        # Reset mesh coordinates after successful evaluation
        if opt['mesh_file_type'] == 'bin':
            dataset.reset_mesh_coordinates()

    if successful_evals == 0:
        print("Warning: No successful evaluations completed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(results)
    df_time = pd.DataFrame(times)
    df_metrics = pd.DataFrame(metrics)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.describe())
    print(df_time.describe())
    print(df_metrics.describe())

    return df, df_time, df_metrics


def firedrake_mesh_to_pyvista(mesh):

    # Get node coordinates and cell connectivity
    coords = mesh.coordinates.dat.data_ro.copy()
    cell_nodes = mesh.coordinates.cell_node_map().values
    num_cells = cell_nodes.shape[0]
    num_nodes_per_cell = cell_nodes.shape[1]  # Should be 3 for triangles

    # For PyVista, create the 'cells' array in the format:
    # [num_nodes_per_cell, node1, node2, node3, num_nodes_per_cell, node1, ...]
    cells = np.hstack((
        np.full((num_cells, 1), num_nodes_per_cell, dtype=np.int64),
        cell_nodes
    )).flatten()

    # Define cell types: 5 for triangles
    cell_types = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)

    # **Convert 2D coords to 3D by adding a zero z-component**
    num_points = coords.shape[0]
    coords_3d = np.zeros((num_points, 3))
    coords_3d[:, :2] = coords  # Set x and y, leave z as zero

    # Create the UnstructuredGrid
    pv_mesh = pv.UnstructuredGrid(cells, cell_types, coords_3d)
    return pv_mesh


def firedrake_mesh_to_pyvista3d(mesh):
    # Get node coordinates and cell connectivity
    coords = mesh.coordinates.dat.data_ro.copy()
    cell_nodes = mesh.coordinates.cell_node_map().values
    num_cells = cell_nodes.shape[0]
    num_nodes_per_cell = cell_nodes.shape[1]  # 3 for triangles, 4 for tetrahedra

    # Detect mesh type
    if num_nodes_per_cell == 3:
        cell_type = pv.CellType.TRIANGLE  # 2D
    elif num_nodes_per_cell == 4:
        cell_type = pv.CellType.TETRA  # 3D
    else:
        raise ValueError(f"Unsupported cell type with {num_nodes_per_cell} nodes per element.")

    # For PyVista, create the 'cells' array:
    # [num_nodes_per_cell, node1, node2, node3, ...]
    cells = np.hstack((
        np.full((num_cells, 1), num_nodes_per_cell, dtype=np.int64),
        cell_nodes
    )).flatten()

    # Set correct cell types
    cell_types = np.full(num_cells, cell_type, dtype=np.uint8)

    # Ensure coordinates are 3D
    num_points = coords.shape[0]
    if coords.shape[1] == 2:  # Convert 2D to 3D if necessary
        coords_3d = np.zeros((num_points, 3))
        coords_3d[:, :2] = coords
    else:
        coords_3d = coords  # Already 3D

    # Create the UnstructuredGrid
    pv_mesh = pv.UnstructuredGrid(cells, cell_types, coords_3d)
    return pv_mesh


def calculate_mesh_quality_metrics(mesh, mesh_coords, mesh_name):
    """Calculate quality metrics for a given mesh configuration.
    Args:
        mesh: Firedrake mesh object
        mesh_coords: Coordinates for the mesh
        mesh_name: Name prefix for the metrics ('grid', 'MA', or 'MLmodel')
    
    Returns:
        dict: Dictionary of mesh quality metrics with prefixed names
    """
    update_mesh_coords(mesh, mesh_coords)
    if mesh_coords.shape[1]==2:
        pv_mesh = firedrake_mesh_to_pyvista(mesh)
    elif mesh_coords.shape[1]==3:
        pv_mesh = firedrake_mesh_to_pyvista3d(mesh)
    metrics = {}
    
    for metric in ['min_angle', 'aspect_ratio', 'scaled_jacobian']:
        quality_mesh = pv_mesh.compute_cell_quality(quality_measure=metric)
        quality_values = quality_mesh['CellQuality']
        
        value = float(np.max(quality_values) if metric == 'aspect_ratio' else np.min(quality_values))
        metrics[f'{mesh_name}_{metric}'] = value
    
    return metrics

