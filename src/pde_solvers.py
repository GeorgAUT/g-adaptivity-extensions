import torch
from firedrake import *
from firedrake.adjoint import *
from firedrake.ml.pytorch import from_torch, fem_operator
from firedrake.__future__ import interpolate

from pde_data import get_pde_params_sample, get_pde_data_sample


def eval_firedrake_fct(uu, u_true, p=2):
    # Note this function is also defined in utils_eval.py, but need to redefine here to avoid circular import
    # Compute and return L1 / L2 error
    if p==1:
        Lp = assemble(sqrt(inner(uu - u_true, uu - u_true)) * dx)
    elif p==2:
        Lp = assemble(inner(uu - u_true, uu - u_true) * dx)
    return Lp


def get_solve_firedrake_class(opt):
    #gets a function that
    #accepts: mesh, c_list, s_list, num_gaussians=1, rand_gaussians=False, bc_type="u_ref"
    # accepts: mesh: , pde_params: {}
    # returns: uu

    # Check if opt['mesh_dims'] is a parameter if not set to 2
    if 'mesh_dims' not in opt:
        dim = 2
    else:
        dim = len(opt['mesh_dims'])

    if dim==1 and opt['pde_type'] == 'Poisson':
        PDESolver = PoissonPDESolver
    elif dim==2 and opt['pde_type'] == 'Poisson':
        PDESolver = PoissonPDESolver
    elif dim==3 and opt['pde_type'] == 'Poisson':
        PDESolver = PoissonPDESolver
    elif dim == 2 and opt['pde_type'] == 'Burgers':
        PDESolver = BurgersPDESolver
    elif dim==2 and opt['pde_type'] == 'NavierStokes':
        PDESolver = NavierStokesPDESolver

    return PDESolver

class PDESolver():
    def __init__(self, opt, dim, mesh_f):

        self.opt = opt
        self.dim = dim
        self.mesh_f = mesh_f
        self.solver = None
        self.u = None
        self.p = None
        self.file = None
        self.z = None
        self.mix_num_gauss = 1 #dummy value
        self.init_solver()

    def init_solver(self):
        pass

    # def get_pde_sample(self):
    #     pass

    def get_pde_function_space(self):
        pass

    def get_pde_data(self):
        # gen_pde_data this has to return an analytic / symbollic function..
        pass
        
    def _get_solver_parameters(self, use_mpi_optimized=False):
        """Get solver parameters based on optimization flag
        
        Args:
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters
            
        Returns:
            dict: Solver parameters dictionary
        """
        if use_mpi_optimized:
            # Use field-split preconditioner for better parallelism
            return {
                'snes_type': 'newtonls',
                'snes_linesearch_type': 'basic',
                'snes_rtol': 1e-8,
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit_0_ksp_type': 'preonly',
                'fieldsplit_0_pc_type': 'lu',
                'ksp_rtol': 1e-8
            }
        else:
            # Default solver parameters
            return {
                'snes_type': 'newtonls',
                'snes_linesearch_type': 'basic',
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'ksp_rtol': 1e-10,
                'ksp_atol': 1e-10
            }

    def update_solver(self, pde_params=None, use_mpi_optimized=False):
        """Base method for updating the solver with optimization parameters
        
        Args:
            pde_params: PDE parameters (if applicable)
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters
        """
        pass

    def update_mesh(self, new_coords):
        self.mesh_f.coordinates.assign(new_coords)

    def solve(self, use_mpi_optimized=False):
        """Solve the PDE problem
        
        Args:
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters.
        
        Returns:
            Function: The solution function
        """
        # Just solve the problem, update_solver should be called explicitly before this
        self.solver.solve()
        return self.z

    def get_Hessian_Frob_norm(self):
        raise NotImplementedError("Hessian recovery is not implemented for this solver class")

    def store(self):
        self.file.write(self.u, self.p)

    def solve_with_mesh_coords(self, mesh_coords, use_mpi_optimized=False):
        """
        Solve the PDE for the current mesh configuration.

        Args:
            mesh_coords: Firedrake Function representing the deformed mesh coordinates.
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters.

        Returns:
            solution: Firedrake Function representing the PDE solution.
        """
        # Update mesh with the provided coordinates
        self.update_mesh(mesh_coords)

        # Solve the PDE and return the solution
        return self.solve(use_mpi_optimized=use_mpi_optimized)

    def loss(self, opt, uu_ref, x_phys):
        """
        Compute the loss for the current PDE setup using Firedrake's adjoint framework.

        Args:
            opt: Dictionary of solver options.
            data: Data object containing PDE parameters and reference solution.
            x_phys: PyTorch tensor of physical mesh coordinates.

        Returns:
            loss: Computed loss value as a PyTorch tensor.
        """
        # Get MPI optimization setting from options
        use_mpi_optimized = opt['use_mpi_optimized']
        
        with stop_annotating():
            # Create a custom tape that is used only in this one instance and then resets
            mytape=Tape()

            with set_working_tape(mytape):
                continue_annotation()

                # Create function space for coordinates and solution
                V_coords = self.mesh_f.coordinates.function_space()
                mesh_coords = Function(V_coords)
                V_HO = FunctionSpace(self.mesh_f, "CG", opt["HO_degree"])

                # Convert PyTorch tensor to Firedrake Function and assign to mesh
                x_phys_double = x_phys.to(dtype=torch.float64)
                x_phys_flat = x_phys_double.reshape(-1)
                deformed_func = from_torch(x_phys_flat, V_coords)
                mesh_coords.assign(deformed_func)

                # Solve PDE and compute reference solution
                u_sol = self.solve_with_mesh_coords(mesh_coords, use_mpi_optimized=use_mpi_optimized)
                with stop_annotating():
                    u_ref = assemble(interpolate(uu_ref, V_HO))

                # Compute the L2 error
                L2_error = eval_firedrake_fct(u_sol, u_ref, p=2)

                # Create reduced functional and PyTorch-compatible operator
                F = ReducedFunctional(L2_error, Control(mesh_coords))

        # After reduced functional is created can stop taping
        loss_operator = fem_operator(F)

        # Compute correction term since interpolate does not keep track of the mesh point dependencies in u_ref
        W = TestFunction(V_coords)
        correction = assemble(-2 * (u_sol - u_ref) * dot(grad(u_ref), W) * dx)
        correction_tensor = torch.tensor(correction.vector().array()).detach()
        # Compute loss using the torch operator
        loss = loss_operator(x_phys_flat)

        # Compute pseudo loss to include the correction term in the gradient
        # Note the numerical value of pseudo_loss has no meaning, but its gradient is correct
        pseudo_loss = loss + torch.sum(x_phys_flat * correction_tensor)


        return pseudo_loss, loss.detach()


class PoissonPDESolver(PDESolver):
    def __init__(self, opt, dim, mesh_f):
        """
        Initialize the Poisson PDE solver.

        Args:
            opt (dict): Dictionary containing solver options.
            dim (int): Dimension of the problem.
            mesh_f (Mesh): Firedrake mesh object.
        """
        super().__init__(opt, dim, mesh_f)

    def get_pde_function_space(self, degree = 1):
        """
        Get the function space for the PDE solution.

        Returns:
            FunctionSpace: Firedrake function space for the solution.
        """
        V = FunctionSpace(self.mesh_f, "CG", degree)  # Function space for the solution
        return V

    def init_solver(self):
        """
        Initialize the solver by setting up the PDE problem.
        """
        pde_params = self.get_pde_params(idx=0, num_data=1, num_gaus=1)

        # Extract Gaussian parameters and convert to Python floats
        #todo generqalise this to work with multiple dimensions
        if isinstance(pde_params['centers'][0], list):
            # self.c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers'][0]]
            # self.s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales'][0]]
            self.c_list = [[float(c[i]) for i in range(self.dim)] for c in pde_params['centers'][0]]
            self.s_list = [[float(s[i]) for i in range(self.dim)] for s in pde_params['scales'][0]]

        else:
            # self.c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers']]
            # self.s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales']]
            self.c_list = [[float(c[i]) for i in range(self.dim)] for c in pde_params['centers']]
            self.s_list = [[float(s[i]) for i in range(self.dim)] for s in pde_params['scales']]

        self.V = self.get_pde_function_space()
        self.bc = None  # Boundary conditions
        self.F_sym = None  # Source term

        self.pde_data = self.get_pde_data(pde_params) # True solution and forcing

        # Define the solution and test functions
        self.z = Function(self.V, name="Solution")
        self.w = TestFunction(self.V)

        # BC and forcing set to zero initially to catch if we forget to update_solver()
        #self.F_form = inner(grad(self.z), grad(self.w)) * dx - self.pde_data['f'] * self.w * dx
        self.F_form = inner(grad(self.z), grad(self.w)) * dx
        #self.bc = DirichletBC(self.V, self.pde_data['u_bc'], "on_boundary")
        self.bc = DirichletBC(self.V, 0, "on_boundary")

        # todo is there a way to not have to initialise the solver again?
        # self.problem = NonlinearVariationalProblem(self.F_form, self.z, bcs=[self.bc])
        # self.solver = NonlinearVariationalSolver(self.problem)

    def get_pde_params(self, idx, num_data, num_gaus=None):
        """
        Get the PDE parameters for the given index and number of data points.

        Args:
            idx (int): Index for the data sample.
            num_data (int): Number of data points.

        Returns:
            dict: Dictionary containing PDE parameters.
        """
        pde_params = get_pde_params_sample(self.opt, self.dim, idx, num_data, num_gaus)
        return pde_params

    def get_pde_data(self, pde_params):
        """
        Get the true solution and source term for the PDE.

        Args:
            pde_params (dict): Dictionary containing PDE parameters.

        Returns:
            tuple: True solution and source term as Firedrake functions.
        """
        x = SpatialCoordinate(self.mesh_f)
        pde_data = get_pde_data_sample(self.opt, x, self.dim, pde_params)
        return pde_data

    # def get_pde_u_true(self, pde_params):
    #     """
    #     Get the true solution for the PDE.
    #
    #     Args:
    #         pde_params (dict): Dictionary containing PDE parameters.
    #
    #     Returns:
    #         Function: True solution as a Firedrake function.
    #     """
    #     U, F = self.get_pde_data(pde_params)
    #     return U
    #
    # def get_pde_f(self, pde_params):
    #     """
    #     Get the source term for the PDE.
    #
    #     Args:
    #         pde_params (dict): Dictionary containing PDE parameters.
    #
    #     Returns:
    #         Function: Source term as a Firedrake function.
    #     """
    #     U, F = self.get_pde_data(pde_params)
    #     return F

    def get_Hessian_Frob_norm(self):
        with stop_annotating():
            Hessian_squared = 0
            n = FacetNormal(self.mesh_f)
            for i in range(self.dim):
                for j in range(self.dim):
                    u_ij = Function(self.V)
                    v = TestFunction(self.V)
                    w = TrialFunction(self.V)
                    solve(
                        w * v * dx == -outer(grad(self.z), grad(v))[i, j] * dx + (outer(n, grad(self.z)) * v)[i, j] * ds,
                        u_ij, bcs=[DirichletBC(self.V, 0, "on_boundary")],
                    )

                    Hessian_squared += u_ij ** 2

            self.Hessian_frob = Function(self.V, name="||H||_F").project(sqrt(Hessian_squared))

        return self.Hessian_frob

    def update_bcs(self, u_ref):
        """
        Update the boundary conditions with the new true solution.

        Args:
            u_ref (Function): True solution as a symbolic function
        """
        raise NotImplementedError("Updating BCs is yet to be implemented.")
        #self.bc.function_space().assign(u_ref)
        # Note u_ref is no longer symbolic, need to pass pde_data here...

    def update_solver(self, pde_params, use_mpi_optimized=False):
        """
        Update the solver with new PDE parameters.

        Args:
            pde_params (dict): Dictionary containing new PDE parameters.
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters.
        """
        # Process new PDE parameters
        self.pde_data = self.get_pde_data(pde_params)

        # Update boundary conditions
        self.bc = DirichletBC(self.V, self.pde_data['u_bc'], "on_boundary")
        #self.problem.bcs = [self.bc]

        #self.problem = NonlinearVariationalProblem(self.F_form, self.z, bcs=[self.bc])
        #self.solver = NonlinearVariationalSolver(self.problem)

        # # Update the variational form by replacing the forcing term
        self.F_form = inner(grad(self.z), grad(self.w)) * dx - self.pde_data['f'] * self.w * dx
        #self.problem.F = self.F_form #inner(grad(self.z), grad(self.w)) * dx - self.pde_data['f'] * self.w * dx
        
        # Get solver parameters from base class method
        solver_params = self._get_solver_parameters(use_mpi_optimized)
            
        #todo is there a way to not have to initialise the solver again?
        self.problem = NonlinearVariationalProblem(self.F_form, self.z, bcs=[self.bc])
        self.solver = NonlinearVariationalSolver(self.problem, solver_parameters=solver_params)


class BurgersPDESolver(PDESolver):
    def __init__(self, opt, dim, mesh_f):
        """
        Initialize the Burgers PDE solver.

        Args:
            opt (dict): Dictionary containing solver options.
            dim (int): Dimension of the problem.
            mesh_f (Mesh): Firedrake mesh object.
        """
        super().__init__(opt, dim, mesh_f)

    def init_solver(self):
        self.V = self.get_pde_function_space()

    def get_pde_function_space(self, degree=1):
        """
        Get the function space for the PDE solution.

        Returns:
            FunctionSpace: Firedrake function space for the solution.
        """
        V = VectorFunctionSpace(self.mesh_f, "CG", degree)  # Function space for the solution
        return V


    def get_pde_params(self, idx, num_data, num_gaus=None):
        """
        Get the PDE parameters for the given index and number of data points.

        Args:
            idx (int): Index for the data sample.
            num_data (int): Number of data points.

        Returns:
            dict: Dictionary containing PDE parameters.
        """
        pde_params = get_pde_params_sample(self.opt, self.dim, idx, num_data, num_gaus)

        # Extract some PDE specific parameters:
        pde_params['amplitude_rescale']=self.opt['amplitude_rescale']
        return pde_params

    def get_pde_data(self, pde_params):
        """
        Get the true solution and source term for the PDE.

        Args:
            pde_params (dict): Dictionary containing PDE parameters.

        Returns:
            tuple: True solution and source term as Firedrake functions.
        """
        x = SpatialCoordinate(self.mesh_f)
        pde_data = get_pde_data_sample(self.opt, x, pde_params)
        return pde_data

    def update_solver(self, pde_params, use_mpi_optimized=False):
        """
        Update the solver with new PDE parameters, actually in the current implementation also initialises the solver.

        Args:
            pde_params (dict): Dictionary containing new PDE parameters.
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters.
        """
        # Process new PDE parameters
        self.pde_data = self.get_pde_data(pde_params)

        # Extract parameters relevant to simulation
        self.nu = self.opt['nu']
        self.timestep = self.opt['timestep']
        self.Nstep = self.opt['num_time_steps']

        # Next we initialise the solution functions and test functions
        self.u_ = Function(self.V, name="Velocity")
        self.u = Function(self.V, name="VelocityNext")

        self.v = TestFunction(self.V)

        # For this problem we need an initial condition::

        #self.ic = project(self.pde_data['u_ic'], self.V) ##

        # We start with current value of u set to the initial condition, but we
        # also use the initial condition as our starting guess for the next
        # value of u::

        # Because we do an assign at the solver call we do not need to assign values for u, u_ here
        #self.u_.assign(self.ic) ##
        #self.u.assign(self.ic) ##

        # Residual of the equation

        self.F_form = (inner((self.u - self.u_) / self.timestep, self.v) + inner(dot(self.u, nabla_grad(self.u)), self.v) + self.nu * inner(grad(self.u), grad(self.v))) * dx

        # Get solver parameters from base class method
        solver_params = self._get_solver_parameters(use_mpi_optimized)
        
        # Create problem and solver with optimized parameters
        self.problem = NonlinearVariationalProblem(self.F_form, self.u)
        self.solver = NonlinearVariationalSolver(self.problem, solver_parameters=solver_params)

    def solve(self):
        # Now solve nonlinear system
        steps = 0

        # Need to project initial condition to track meshpoint dependency
        f=self.pde_data['u_ic']
        solve(inner((self.u_-f),self.v)*dx==0,self.u_)

        # Initialise u next
        self.u.assign(self.u_)

        # Redefine the nonlinear form (may not be necessary)
        #self.F_form = (inner((self.u - self.u_) / self.timestep, self.v) + inner(dot(self.u, nabla_grad(self.u)), self.v) + self.nu * inner(grad(self.u),grad(self.v))) * dx

        while (steps < self.Nstep):
            self.solver.solve()
            self.u_.assign(self.u)
            steps += 1
        return self.u

    def solve_with_given_ic(self, u_ic):
        # Now solve nonlinear system
        steps = 0
        # Need to project initial condition to track meshpoint dependency
        solve(inner((self.u_ - u_ic), self.v) * dx == 0, self.u_)
        # Initialise u next
        self.u.assign(self.u_)
        # Redefine the nonlinear form (may not be necessary)
        # self.F_form = (inner((self.u - self.u_) / self.timestep, self.v) + inner(dot(self.u, nabla_grad(self.u)), self.v) + self.nu * inner(grad(self.u),grad(self.v))) * dx
        while (steps < self.Nstep):
            self.solver.solve()
            self.u_.assign(self.u)
            steps += 1
        return self.u

    def get_Hessian_Frob_norm(self):
        with stop_annotating():
            Hessian_squared = 0
            n = FacetNormal(self.mesh_f)
            V = FunctionSpace(self.mesh_f, "CG", 1)
            for l in range(2):
                ul = assemble(interpolate(self.u[l], V))
                for i in range(self.dim):
                    for j in range(self.dim):
                        u_ij = Function(V)
                        v = TestFunction(V)
                        w = TrialFunction(V)
                        solve(
                            w * v * dx == -outer(grad(ul), grad(v))[i, j] * dx + (outer(n, grad(ul)) * v)[i, j] * ds,
                            u_ij, bcs=[DirichletBC(V, 0, "on_boundary")],
                        )

                        Hessian_squared += u_ij ** 2

            self.Hessian_frob = Function(V, name="||H||_F").project(sqrt(Hessian_squared))

        return self.Hessian_frob

    def loss(self, opt, uu_ref, x_phys):
        """
        Compute the loss for the current PDE setup using Firedrake's adjoint framework. Burgers requires some minor adjustments hence redefined over function class

        Args:
            opt: Dictionary of solver options.
            data: Data object containing PDE parameters and reference solution.
            x_phys: PyTorch tensor of physical mesh coordinates.

        Returns:
            loss: Computed loss value as a PyTorch tensor.
            pseudo_loss: Quantity that needs to be differentiated for correct gradient computation.
        """
        with stop_annotating():
            # Create a custom tape that is used only in this one instance and then resets
            mytape=Tape()

            with set_working_tape(mytape):
                continue_annotation()

                # Create function space for coordinates and solution
                V_coords = self.mesh_f.coordinates.function_space()
                mesh_coords = Function(V_coords)
                V_HO = VectorFunctionSpace(self.mesh_f, "CG", opt["HO_degree"])

                # Convert PyTorch tensor to Firedrake Function and assign to mesh
                x_phys_double = x_phys.to(dtype=torch.float64)
                x_phys_flat = x_phys_double.reshape(-1)
                deformed_func = from_torch(x_phys_flat, V_coords)
                mesh_coords.assign(deformed_func)

                # Set up tape and create torch operator for PDE solve
                #with (set_working_tape() as tape):
                # Solve PDE and compute reference solution
                u_sol = self.solve_with_mesh_coords(mesh_coords)
                with stop_annotating():
                    u_ref = assemble(interpolate(uu_ref, V_HO))

                # Compute the L2 error
                L2_error = eval_firedrake_fct(u_sol, u_ref, p=2)

                # Create reduced functional and PyTorch-compatible operator
                F = ReducedFunctional(L2_error, Control(mesh_coords))

        # After reduced functional is created can stop taping
        loss_operator = fem_operator(F)

        # Compute correction term since interpolate does not keep track of the mesh point dependencies in u_ref
        W = TestFunction(V_coords)
        aux=-2*(self.u-u_ref)
        #correction = assemble(inner(aux,dot(grad(u_ref))))
        correction = assemble(aux[0]*dot(grad(u_ref[0]),  W)*dx+aux[1]*dot(grad(u_ref[1]),  W)*dx)
        correction_tensor = torch.tensor(correction.vector().array()).detach()
        # Compute loss using the torch operator
        loss = loss_operator(x_phys_flat)

        # Compute pseudo loss to include the correction term in the gradient
        # Note the numerical value of pseudo_loss has no meaning, but its gradient is correct
        pseudo_loss = loss + torch.sum(x_phys_flat * correction_tensor)

        # # Test the gradient computation
        # x_phys_flat1 = x_phys_flat.clone().detach().requires_grad_(True)
        # x_phys_flat2 = x_phys_flat.clone().detach().requires_grad_(True)
        # loss1 = loss_operator(x_phys_flat1)
        # loss = loss_operator(x_phys_flat2) + torch.sum(x_phys_flat2 * correction_tensor)
        #
        # loss1.backward()
        # loss.backward()
        #
        # print(torch.linalg.norm(x_phys_flat1.grad - x_phys_flat2.grad))
        #
        # print("The above is just for testing")
        return pseudo_loss, loss.detach()


class NavierStokesPDESolver(PDESolver):
    """Incompressible Stokes solver with Taylor-Hood elements."""
    def __init__(self, opt, dim, mesh_f):
        """
        Initialize the NavierStokes PDE solver.

        Args:
            opt (dict): Dictionary containing solver options.
            dim (int): Dimension of the problem.
            mesh_f (Mesh): Firedrake mesh object.
        """
        super().__init__(opt, dim, mesh_f)

    def init_solver(self):
        self.V, self.Q = self.get_pde_function_space()

    def get_pde_function_space(self, degree=1):
        """
        Get the function space for the PDE solution.

        Returns:
            FunctionSpace: Firedrake function space for the solution.
        """
        # Taylor-Hood elements
        V = VectorFunctionSpace(self.mesh_f, "CG", degree + 1) # velocity space
        Q = FunctionSpace(self.mesh_f, "CG", degree) # pressure space
        return V, Q

    def get_pde_params(self, idx, num_data, num_gaus=1):
        """
        Get the PDE parameters for the given index and number of data points.

        Args:
            idx (int): Index for the data sample.
            num_data (int): Number of data points.

        Returns:
            dict: Dictionary containing PDE parameters.
        """
        pde_params = get_pde_params_sample(self.opt, self.dim, idx, num_data, num_gaus)

        # Extract some PDE specific parameters:
        return pde_params

    def get_pde_data(self, pde_params):
        """
        Get the true solution and source term for the PDE.

        Args:
            pde_params (dict): Dictionary containing PDE parameters.

        Returns:
            tuple: True solution and source term as Firedrake functions.
        """
        x = SpatialCoordinate(self.mesh_f)
        pde_data = get_pde_data_sample(self.opt, x, pde_params)
        return pde_data

    def update_solver(self, pde_params, use_mpi_optimized=False):
        """
        Update the solver with new PDE parameters, actually in the current implementation also initialises the solver.

        Args:
            pde_params (dict): Dictionary containing new PDE parameters.
            use_mpi_optimized (bool): If True, use MPI-optimized solver parameters.
        """
        # Process new PDE parameters
        self.pde_data = self.get_pde_data(pde_params)

        # Extract parameters relevant to simulation
        self.nu_val = self.opt['nu']
        self.nu = Constant(self.nu_val)
        self.dt = self.opt['timestep']
        # define a firedrake constant equal to dt so that variation forms
        # not regenerated if we change the time step
        self.k = Constant(self.dt)

        self.Nstep = self.opt['num_time_steps']
        self.U_mean = self.opt['U_mean']



        # Velocity test and trial functions
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Pressure test and trial functions
        self.p = TrialFunction(self.Q)
        self.q = TestFunction(self.Q)

        self.u_now = Function(self.V)
        self.u_next = Function(self.V)
        self.u_star = Function(self.V)
        self.p_now = Function(self.Q)
        self.p_next = Function(self.Q)

        # Expressions for the variational forms
        self.n = FacetNormal(self.mesh_f)
        self.f = Constant((0.0, 0.0))
        self.u_mid = 0.5 * (self.u_now + self.u)

        def sigma(u, p):
            return 2 * self.nu * sym(nabla_grad(u)) - p * Identity(len(u))

        x, y = SpatialCoordinate(self.mesh_f)

        # Define boundary conditions
        self.bcu = [
            DirichletBC(self.V, Constant((0, 0)), (1, 4)),  # top-bottom and cylinder
            DirichletBC(self.V, ((4.0 * 1.5 * y * (0.41 - y) / 0.41 ** 2), 0), 2),
        ]  # inflow
        self.bcp = [DirichletBC(self.Q, Constant(0), 3)]  # outflow

        self.re_num = int(self.U_mean * 0.1 / self.nu_val)  # Reynolds number
        # print(f"Re = {re_num}")

        # Define variational forms
        self.F1 = (
                inner((self.u - self.u_now) / self.k, self.v) * dx
                + inner(dot(self.u_now, nabla_grad(self.u_mid)), self.v) * dx
                + inner(sigma(self.u_mid, self.p_now), sym(nabla_grad(self.v))) * dx
                + inner(self.p_now * self.n, self.v) * ds
                - inner(self.nu * dot(nabla_grad(self.u_mid), self.n), self.v) * ds
                - inner(self.f, self.v) * dx
        )

        self.a1, self.L1 = system(self.F1)

        self.a2 = inner(nabla_grad(self.p), nabla_grad(self.q)) * dx
        self.L2 = (
                inner(nabla_grad(self.p_now), nabla_grad(self.q)) * dx
                - (1 / self.k) * inner(div(self.u_star), self.q) * dx
        )

        self.a3 = inner(self.u, self.v) * dx
        self.L3 = (
                inner(self.u_star, self.v) * dx - self.k * inner(nabla_grad(self.p_next - self.p_now), self.v) * dx
        )

        # Define linear problems
        self.problem1 = LinearVariationalProblem(self.a1, self.L1, self.u_star, bcs=self.bcu)
        self.problem2 = LinearVariationalProblem(self.a2, self.L2, self.p_next, bcs=self.bcp)
        self.problem3 = LinearVariationalProblem(self.a3, self.L3, self.u_next)

        # Define solvers with optimization parameters if requested
        if use_mpi_optimized:
            # MPI-optimized solver parameters
            solver_params1 = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit_0_ksp_type': 'preonly',
                'fieldsplit_0_pc_type': 'lu',
                'ksp_rtol': 1e-8
            }
            solver_params2 = {
                'ksp_type': 'cg',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit_0_pc_type': 'gamg',
                'ksp_rtol': 1e-8
            }
            solver_params3 = {
                'ksp_type': 'cg',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'additive',
                'fieldsplit_0_pc_type': 'sor',
                'ksp_rtol': 1e-8
            }
            
            self.solver1 = LinearVariationalSolver(self.problem1, solver_parameters=solver_params1)
            self.solver2 = LinearVariationalSolver(self.problem2, solver_parameters=solver_params2)
            self.solver3 = LinearVariationalSolver(self.problem3, solver_parameters=solver_params3)
        else:
            # Use original default solver parameters
            self.solver1 = LinearVariationalSolver(
                self.problem1, solver_parameters={"ksp_type": "gmres", "pc_type": "sor"}
            )
            self.solver2 = LinearVariationalSolver(
                self.problem2, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"}
            )
            self.solver3 = LinearVariationalSolver(
                self.problem3, solver_parameters={"ksp_type": "cg", "pc_type": "sor"}
            )


    def solve(self):

        # Now solve nonlinear system
        steps = 0

        # Need to project initial condition to track meshpoint dependency
        u_ic_sym = self.pde_data['u_ic']
        p_ic_sym = self.pde_data['p_ic']
        solve(inner((self.u_now - u_ic_sym), self.v) * dx == 0, self.u_now)
        solve(inner((self.p_now - p_ic_sym), self.q) * dx == 0, self.p_now)



        # Initialise u next
        self.u_star.assign(self.u_now)
        self.u_next.assign(self.u_now)
        self.p_next.assign(self.p_now)

        while (steps < self.Nstep):
            self.solver1.solve()
            self.solver2.solve()
            self.solver3.solve()

            # update solutions
            self.u_now.assign(self.u_next)
            self.p_now.assign(self.p_next)


            steps += 1

        return self.u_now, self.p_now

    def get_Hessian_Frob_norm(self):
        with stop_annotating():
            Hessian_squared = 0
            n = FacetNormal(self.mesh_f)
            V = FunctionSpace(self.mesh_f, "CG", 1)
            for l in range(2):
                ul = assemble(interpolate(self.u_now[l], V))
                for i in range(self.dim):
                    for j in range(self.dim):
                        u_ij = Function(V)
                        v = TestFunction(V)
                        w = TrialFunction(V)
                        solve(
                            w * v * dx == -outer(grad(ul), grad(v))[i, j] * dx + (outer(n, grad(ul)) * v)[i, j] * ds,
                            u_ij, bcs=[DirichletBC(V, 0, "on_boundary")],
                        )

                        Hessian_squared += u_ij ** 2

            for i in range(self.dim):
                for j in range(self.dim):
                    p_ij = Function(V)
                    v = TestFunction(V)
                    w = TrialFunction(V)
                    solve(
                        w * v * dx == -outer(grad(self.p_now), grad(v))[i, j] * dx + (outer(n, grad(self.p_now)) * v)[i, j] * ds,
                        p_ij, bcs=[DirichletBC(V, 0, "on_boundary")],
                    )

                    Hessian_squared += p_ij ** 2

            self.Hessian_frob = Function(V, name="||H||_F").project(sqrt(Hessian_squared))

        return self.Hessian_frob

    # Note the below is a slightly different way for Hessian recovery since here have CG 2 elements for u. But for the time being the above might be acceptable.
    # A, B = PDESolver_coarse.get_pde_function_space()
    # H_test1 = Function(B)
    # H_test2 = Function(B)
    # H_test3 = Function(B)
    # H_test4 = Function(B)
    # v = TestFunction(B)
    # trial = TrialFunction(B)
    # solve(trial * v * dx == grad(grad(u[0])[0])[0] * v * dx, H_test1)
    # solve(trial * v * dx == grad(grad(u[0])[1])[1] * v * dx, H_test2)
    # solve(trial * v * dx == grad(grad(u[1])[0])[0] * v * dx, H_test3)
    # solve(trial * v * dx == grad(grad(u[1])[1])[1] * v * dx, H_test4)
    #
    # H_test = assemble(interpolate(sqrt(H_test1 ** 2 + H_test2 ** 2 + H_test3 ** 2 + H_test4 ** 2), B))
    # # -outer(grad(self.p_now), grad(v))[i, j] * dx + (outer(n, grad(self.p_now)) * v)[i, j] * ds,
    # #                     p_ij, bcs=[DirichletBC(V, 0, "on_boundary")],

    def loss(self, opt, up_ref, x_phys):
        """
        Compute the loss for the current PDE setup using Firedrake's adjoint framework. NavierStokes requires some minor adjustments hence redefined over function class

        Args:
            opt: Dictionary of solver options.
            data: Data object containing PDE parameters and reference solution.
            x_phys: PyTorch tensor of physical mesh coordinates.

        Returns:
            loss: Computed loss value as a PyTorch tensor.
            pseudo_loss: Quantity that needs to be differentiated for correct gradient computation.
        """

        # Extract reference solution values
        uu_ref, pp_ref = up_ref

        with stop_annotating():
            # Create a custom tape that is used only in this one instance and then resets
            mytape=Tape()

            with set_working_tape(mytape):
                continue_annotation()

                # Create function space for coordinates and solution
                V_coords = self.mesh_f.coordinates.function_space()
                mesh_coords = Function(V_coords)
                V_HO = VectorFunctionSpace(self.mesh_f, "CG", opt["HO_degree"]+1)
                Q_HO = FunctionSpace(self.mesh_f, "CG", opt["HO_degree"])

                # Convert PyTorch tensor to Firedrake Function and assign to mesh
                x_phys_double = x_phys.to(dtype=torch.float64)
                x_phys_flat = x_phys_double.reshape(-1)
                deformed_func = from_torch(x_phys_flat, V_coords)
                mesh_coords.assign(deformed_func)

                # Set up tape and create torch operator for PDE solve
                #with (set_working_tape() as tape):
                # Solve PDE and compute reference solution
                u_sol, p_sol = self.solve_with_mesh_coords(mesh_coords)
                with stop_annotating():
                    u_ref = assemble(interpolate(uu_ref, V_HO))
                    p_ref = assemble(interpolate(pp_ref, Q_HO))

                # Compute the L2 error
                L2_error = eval_firedrake_fct(u_sol, u_ref, p=2) + eval_firedrake_fct(p_sol, p_ref, p=2)

                # Create reduced functional and PyTorch-compatible operator
                F = ReducedFunctional(L2_error, Control(mesh_coords))

        # After reduced functional is created can stop taping
        loss_operator = fem_operator(F)

        # Compute correction term since interpolate does not keep track of the mesh point dependencies in u_ref
        W = TestFunction(V_coords)

        #correction = assemble(inner(aux,dot(grad(u_ref))))
        correction = assemble(-2*(u_sol[0]-u_ref[0])*dot(grad(u_ref[0]),  W)*dx-2*(u_sol[1]-u_ref[1])*dot(grad(u_ref[1]),  W)*dx-2*(p_sol-p_ref)*dot(grad(p_ref),  W)*dx)
        correction_tensor = torch.tensor(correction.vector().array()).detach()
        # Compute loss using the torch operator
        loss = loss_operator(x_phys_flat)

        # Compute pseudo loss to include the correction term in the gradient
        # Note the numerical value of pseudo_loss has no meaning, but its gradient is correct
        pseudo_loss = loss + torch.sum(x_phys_flat * correction_tensor)

        # # Test the gradient computation
        # x_phys_flat1 = x_phys_flat.clone().detach().requires_grad_(True)
        # x_phys_flat2 = x_phys_flat.clone().detach().requires_grad_(True)
        # loss1 = loss_operator(x_phys_flat1)
        # loss = loss_operator(x_phys_flat2) + torch.sum(x_phys_flat2 * correction_tensor)
        #
        # loss1.backward()
        # loss.backward()
        #
        # print(torch.linalg.norm(x_phys_flat1.grad - x_phys_flat2.grad))
        #
        # print("The above is just for testing")
        return pseudo_loss, loss.detach()



    # def init_solver(self):
    #     # Taylor-Hood elements
    #     Vf_velocity = VectorFunctionSpace(mesh_f, "CG", 2)
    #     Vf_pressure = FunctionSpace(mesh_f, "CG", 1)
    #     Vf = Vf_velocity * Vf_pressure
    #
    #     # weak form
    #     z = Function(Vf)
    #     w = TestFunction(Vf)
    #     u, p = split(z)
    #     v, q = split(w)
    #     nu = 1/10  # viscosity
    #     F = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
    #     # DX = dx(quadrature_degree=2) needs to be higher for error computation..
    #     # F = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx(quadrature_degree=2) check this
    #
    #     # boundary conditions
    #     x, y = SpatialCoordinate(mesh_f)
    #     uin = as_vector([4-y**2, 0])
    #     bc1 = DirichletBC(Vf.sub(0), uin, 12)  # inlet velocity
    #     bc2 = DirichletBC(Vf.sub(0), 0, [11, 13])  # no slip condition
    #     prb = NonlinearVariationalProblem(F, z, bcs=[bc1, bc2])
    #     self.solver = NonlinearVariationalSolver(prb)
    #
    #     # file to store solutions
    #     u_, p_ = z.subfunctions
    #     self.u = u_
    #     self.p = p_
    #     self.file = VTKFile("soln.pvd")
    #     self.z = z #solution
    #
    # def solve(self):
    #     self.solver.solve()
    #
    # def store(self):
    #     self.file.write(self.u, self.p)
    #
    # def update_mesh(self, new_coords):
    #     self.mesh_f.coordinates.assign(new_coords)