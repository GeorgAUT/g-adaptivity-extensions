## This file contains a wrapper to allow for direct movement of meshpoints via gradient descent of the loss
# Here no surrogate is used, and this is not for practical mesh movement

import torch
import time
import torch.nn.functional as F
import numpy as np
from firedrake import *

from pde_solvers import get_solve_firedrake_class
from firedrake_difFEM.difFEM_2d import torch_FEM_2D
from utils_data import reshape_grid_to_fd_tensor
from utils_train import equidistribution_loss

class backFEM_2D(torch.nn.Module):
    '''a wrapper for differentiable backFEM solver that minimizes the L2 error of the approximation and returns the updated mesh, no surrogate here'''

    def __init__(self, opt, dataset=None):
        super().__init__()
        self.opt = opt
        if dataset:
            self.dataset = dataset
        self.num_meshpoints = opt['mesh_dims'][0] #internal mesh points
        self.lr = opt['lr']
        self.epochs = opt['epochs']
        self.loss_list = []
        self.gradx_list = []
        self.grady_list = []


    def forward(self, data):
        self.start_MLmodel = time.time()
        # Initialise PDE solver for meshpoint gradient computation
        self.SolverCoarse = get_solve_firedrake_class(self.opt)
        #n = self.opt['mesh_dims'][0]
        #m = self.opt['mesh_dims'][1]
        self.coarse_mesh = data.coarse_mesh[0]#UnitSquareMesh(n - 1, m - 1, name="coarse_mesh")
        self.PDESolver_coarse = self.SolverCoarse(self.opt, len(self.opt['mesh_dims']), self.coarse_mesh)

        # Allow for use on random dataset
        if self.opt['data_type'] == 'randg_mix':
            pde_params = data.batch_dict[0]['pde_params']
        else:
            pde_params = data.pde_params

        self.PDESolver_coarse.update_solver(pde_params)

        # Create internal mesh points
        # TODO: make this Navier-Stokes compatible
        V = self.PDESolver_coarse.get_pde_function_space()
        bc = DirichletBC(V, 0, "on_boundary") # These BC are not actually used, just for sake of identifying internal nodes
        #bc.nodes  # Get the boundary nodes
        # Index complement of bc.nodes
        internal_ind = np.setdiff1d(np.arange(np.shape(self.coarse_mesh.coordinates.dat.data[:])[0]), bc.nodes)

        # Initialize mesh points - either from regular grid or MA mesh
        if self.opt['start_from_ma'] and hasattr(data, 'x_ma'):
            # Start from MA mesh if available
            initial_coords = data.x_ma.detach().numpy()
            self.coarse_mesh.coordinates.dat.data[:] = initial_coords
            internal_mesh_points = torch.tensor(initial_coords[internal_ind], dtype=torch.float32, requires_grad=True)
            x_phys = torch.tensor(initial_coords, dtype=torch.float32, requires_grad=False)
        else:
            # Start from regular grid (default behavior)
            internal_mesh_points = torch.tensor(self.coarse_mesh.coordinates.dat.data[internal_ind], dtype=torch.float32,
                                             requires_grad=True)
            x_phys = torch.tensor(self.coarse_mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)

        # Define the differentiable mesh points, making sure Boundary nodes are fixed
        x_phys[internal_ind] = internal_mesh_points

        # Use SGD optimizer for gradient descent
        optimizer = torch.optim.SGD([internal_mesh_points], lr=self.opt['lr'])
        if self.opt['loss_fn'] == 'mse':
            loss_fn = F.mse_loss
        elif self.opt['loss_fn'] == 'l1':
            loss_fn = F.l1_loss

        self.loss_list = []
        self.gradx_list = []
        self.grady_list = []
        
        total_nodes = np.shape(self.coarse_mesh.coordinates.dat.data[:])[0]
        for j in range(self.opt['epochs']):
            print(j)
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Compute the loss with the PDE solver

            opt =self.opt
            if opt['loss_type'] == 'mesh_loss':
                data = data.to(opt['device'])
                loss = loss_fn(x_phys, x_phys)
                loss.backward()

            elif opt['loss_type'] == 'pde_loss_torch':
                c_list_torch = [torch.from_numpy(c_0).to(self.opt['device']) for c_0 in pde_params['centers'][0]]
                s_list_torch = [torch.from_numpy(s_0).to(self.opt['device']) for s_0 in pde_params['scales'][0]]
                mesh = self.dataset.coarse_mesh

                num_meshpoints = int(np.sqrt(x_phys.shape[0]))
                x0 = torch.linspace(0, 1, self.opt['eval_quad_points'])  # , dtype=torch.float64)
                y0 = torch.linspace(0, 1, self.opt['eval_quad_points'])  # , dtype=torch.float64)
                [X, Y] = torch.meshgrid(x0, y0, indexing='ij')
                quad_points = [X, Y]
                coeffs, x_phys, sol = torch_FEM_2D(self.opt, mesh, x_phys,
                                                 quad_points=quad_points,
                                                 num_meshpoints=num_meshpoints,
                                                 c_list=c_list_torch, s_list=s_list_torch)
                sol_fd = reshape_grid_to_fd_tensor(sol.view(-1).unsqueeze(-1),
                                                 self.dataset.mapping_tensor_fine)


                uu_ref = data.uu_ref[0]
                u_true_fine_tensor = torch.from_numpy(uu_ref.dat.data[:])
                u_true_fine_tensor = u_true_fine_tensor.float()
                loss = loss_fn(sol_fd.to(opt['device']), u_true_fine_tensor.to(opt['device']))
                loss.backward()

            elif opt['loss_type'] == 'pde_loss_firedrake':
                pseudo_loss, loss = self.PDESolver_coarse.loss(self.opt, data.uu_ref[0], x_phys)
                
                # Add equidistribution loss with weight
                equi_loss = equidistribution_loss(x_phys, data)
                pseudo_loss = pseudo_loss + opt['loss_regulariser_weight'] * equi_loss
                
                pseudo_loss.backward()  # Note this is differentiable wrt meshpoints

            # Store gradients
            grads_x = np.zeros(total_nodes)
            grads_y = np.zeros(total_nodes)
            internal_grads = internal_mesh_points.grad.cpu().numpy()
            grads_x[internal_ind] = internal_grads[:, 0]
            grads_y[internal_ind] = internal_grads[:, 1]
            self.gradx_list.append(grads_x)
            self.grady_list.append(grads_y)

            # Update mesh points
            optimizer.step()

            print("Iteration:", j, "Loss:", loss.item())

            # Redefine full mesh points to ensure boundary nodes are fixed and not differentiating through computational tree twice
            x_phys = torch.tensor(self.coarse_mesh.coordinates.dat.data, dtype=torch.float32, requires_grad=False)
            x_phys[internal_ind] = internal_mesh_points

            optimizer.zero_grad()
            self.loss_list.append(loss.item())


        self.end_MLmodel = time.time()
        return x_phys