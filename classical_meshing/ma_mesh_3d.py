"""Utilities for 3D Monge-Ampère mesh movement."""

from firedrake import *
import movement
import numpy as np
import time
from movement.monge_ampere import *


def deform_mesh_ma3d(x_comp, coarse_mesh, PDESolver_coarse, u, Hessian_Frob_u, opt, pde_params, SolverCoarse):
    x_phys, j, build_time = MA3d_new(x_comp, coarse_mesh, u, Hessian_Frob_u, opt, pde_params, SolverCoarse)
    j=j+1 #to account for the initial mesh
    return x_phys, j, build_time


def MA3d_new(x_comp, mesh, u, Hessian_Frob_u, opt, pde_params, SolverCoarse):
    mesh.coordinates.dat.data[:, :] = x_comp.detach().cpu().numpy()

    def monitor(mesh):
        V = FunctionSpace(mesh, "CG", 1)  # Auxiliary function space required for MA mover when referring to approx solution
        if opt['onitoritor_type'] == 'monitor_hessian_approx':
            maxH=np.max(Hessian_Frob_u.dat.data[:])
            output = Function(V)  # Output, need to reinterpolate the monitor fn to mesh function space
            output.interpolate(Constant(1) + Constant(opt['monitor_alpha']) * Hessian_Frob_u / Constant(maxH))
            print(f"output {sum(output.dat.data)}")
            print(f"mesh {sum(mesh.coordinates.dat.data)}")
            if mesh is None:
                raise ValueError("Mesh input is None. Ensure it is initialized properly.")
        elif opt['monitor_type'] == 'monitor_hessian':
            PDESolver_coarse = SolverCoarse(opt, 3, mesh)
            PDESolver_coarse.update_solver(pde_params)

            u = PDESolver_coarse.solve()

            #save inputs to GNN
            Hessian_Frob_u1 = PDESolver_coarse.get_Hessian_Frob_norm()
            maxH1 = np.max(Hessian_Frob_u1.dat.data[:])

            output = Function(V)  # Output, need to reinterpolate the monitor fn to mesh function space
            output.interpolate(Constant(1) + Constant(opt['monitor_alpha']) * Hessian_Frob_u1 / Constant(maxH1))
        else:
            raise ValueError("The desired MA_monitor is not implemented, please use opt['monitor_type'] == 'monitor_hessian_approx' or 'monitor_hessian'")
        if mesh is None:
            raise ValueError("Mesh input is None. Ensure it is initialized properly.")
        return output

    mover = MongeAmpereMover(mesh, monitor, method = "relaxation", rtol = 1e-12, maxiter = 1000)
    start_time = time.time()

    try:
        j = mover.move()
        x_phys = mover.mesh.coordinates.dat.data[:, :]
    except:
        j = 0
        x_phys = mesh.coordinates.dat.data[:, :]
        Warning('Monge-Ampere solver did not converge (see ma_mesh_2d.py)')

    build_time = time.time() - start_time
    return x_phys, j, build_time