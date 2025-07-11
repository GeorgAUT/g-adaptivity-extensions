import numpy as np
from firedrake import *

from navier_stokes_data.series_approximation import load_expansion


def get_pde_params_sample(opt, dim, idx, num_data, num_gauss):
    """
    Get PDE parameters for a sample based on the PDE type and data type.
    
    Args:
        opt (dict): Options dictionary
        dim (int): Dimension of the problem
        idx (int): Index of the sample
        num_data (int): Total number of data points
        
    Returns:
        dict: PDE parameters
    """
    pde_params = None

    if opt['pde_type'] == 'Poisson':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            pde_params = create_gaussian_params(opt, dim, idx, num_data, num_gauss)
    elif opt['pde_type'] == 'Burgers':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            pde_params = create_gaussian_params(opt, dim, idx, num_data, num_gauss)
            pde_params['amplitude_rescale'] = opt['amplitude_rescale']
    elif opt['pde_type'] == 'NavierStokes':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            pde_params = create_gaussian_params(opt, dim, idx, num_data, num_gauss)
        elif opt['data_type'] == 'RBF':
            pde_params = create_expansion_params(opt, dim, idx, num_data)

    if pde_params is None:
        raise ValueError(f"No PDE parameters could be generated for PDE type '{opt['pde_type']}' and data type '{opt['data_type']}'")

    return pde_params

def get_pde_data_sample(opt, x, dim, pde_params):
    if opt['pde_type'] == 'Poisson':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            return get_pde_data_sample_gaussian_poisson(x, dim, pde_params)
    elif opt['pde_type'] == 'Burgers':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            return get_pde_data_sample_gaussian_burgers(x, pde_params)
    elif opt['pde_type'] == 'NavierStokes':
        if opt['data_type'] in ['randg', 'structured', 'randg_mix']:
            return get_pde_data_sample_gaussian_navierstokes(x, pde_params)
        elif opt['data_type'] == 'RBF':
            return get_pde_data_sample_rbf_navierstokes(x, pde_params)
    else:
        raise ValueError("PDE type not recognized.")

def get_pde_data_sample_gaussian_poisson(x, dim, pde_params):
    # Extract Gaussian parameters and convert to Python floats
    if type(pde_params['centers'][0]) == list:
        c_list = [[float(c[i]) for i in range(dim)] for c in pde_params['centers'][0]]
        s_list = [[float(s[i]) for i in range(dim)] for s in pde_params['scales'][0]]
    else:
        c_list = [[float(c[i]) for i in range(dim)] for c in pde_params['centers']]
        s_list = [[float(s[i]) for i in range(dim)] for s in pde_params['scales']]

    # Construct true solution and source term from Gaussians
    u_true_total = 0.
    F = 0.
    for c, s in zip(c_list, s_list):
        # Compute squared differences normalized by scale parameters
        exp_argument = -sum((x[i] - c[i]) ** 2 / s[i] ** 2 for i in range(dim))

        # True solution as sum of Gaussians
        gaussian = exp(exp_argument)
        # if dim == 3:
        #     gaussian = gaussian * 20.0  # Scale up the solution for 3D Poisson
        # Source term is the Laplacian of the Gaussians

        prefactor = -sum((4*(x[i]-c[i])**2-2*s[i]**2)/s[i]**4 for i in range(dim))
        laplacian = prefactor*gaussian

        u_true_total += gaussian
        F += laplacian
    pde_data = {'u_bc': u_true_total, 'f': F} # u_bc is used to specify boundary conditions only, not as a reference solution
    return pde_data

def get_pde_data_sample_gaussian_burgers(x, pde_params):
    # TODO: Might have to restrict location and scale of Gaussians to avoid illposedness in Burgers
    # Extract Gaussian parameters and convert to Python floats
    if type(pde_params['centers'][0]) == list:
        c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers'][0]]
        s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales'][0]]
    else:
        c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers']]
        s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales']]

    # Construct true solution and source term from Gaussians
    u_ic_total = 0.
    F = 0.

    for c, s in zip(c_list, s_list):
        # True solution is a sum of Gaussians
        gaussian = exp(-(x[0] - c[0]) ** 2 / s[0] ** 2 - (x[1] - c[1]) ** 2 / s[1] ** 2)
        # Source term is the Laplacian of the Gaussians
        u_ic_total += gaussian*pde_params['amplitude_rescale']

    pde_data = {'u_ic': as_vector([u_ic_total, 0])} # u_ic is the initial condition
    return pde_data

def get_pde_data_sample_gaussian_navierstokes(x, pde_params):
    # Note this is just a testing code and not the actual initial condition for Navier-Stokes
    # Extract Gaussian parameters and convert to Python floats
    if type(pde_params['centers'][0]) == list:
        c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers'][0]]
        s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales'][0]]
    else:
        c_list = [[float(c[0]), float(c[1])] for c in pde_params['centers']]
        s_list = [[float(s[0]), float(s[1])] for s in pde_params['scales']]

    # Construct true solution and source term from Gaussians
    u_ic_total = 0.
    F = 0.
    for c, s in zip(c_list, s_list):
        # True solution is a sum of Gaussians
        gaussian = exp(-(x[0] - c[0]) ** 2 / s[0] ** 2 - (x[1] - c[1]) ** 2 / s[1] ** 2)
        # Source term is the Laplacian of the Gaussians
        u_ic_total += gaussian

    pde_data = {'u_ic': as_vector([u_ic_total*0.01, 0]), 'p_ic': gaussian*0.01} # u_ic is the initial condition
    return pde_data

def get_pde_data_sample_rbf_navierstokes(x, pde_params):
    # Note this is just a testing code and not the actual initial condition for Navier-Stokes

    if type(pde_params['filename_u']) == list:
        u_expansion = load_expansion(pde_params['filename_u'][0])
        p_expansion = load_expansion(pde_params['filename_p'][0])
    else:
        u_expansion = load_expansion(pde_params['filename_u'])
        p_expansion = load_expansion(pde_params['filename_p'])
    u_ic = u_expansion.as_expression(x[0], x[1])
    p_ic = p_expansion.as_expression(x[0], x[1])

    pde_data = {'u_ic': u_ic, 'p_ic': p_ic} # u_ic is the initial condition
    return pde_data


def create_gaussian_params(opt: dict, dim: int, idx: int, num_data: int, num_gauss: int = None) -> dict:
    """
    Creates Gaussian parameters for PDE data based on the provided options.

    Args:
        opt (dict): Dictionary containing options for generating Gaussian parameters.
        dim (int): Dimension of the Gaussian parameters.
        idx (int): Index used for structured data generation.
        num_data (int): Total number of data points.

    Returns:
        dict: Dictionary containing the centers and scales of the Gaussian parameters.
    """

    c_list = []
    s_list = []

    if opt['data_type'] in ['randg']:
        mesh_scale = opt.get('mesh_scale', 1.0)  # Default to 1.0 if not specified
        for j in range(opt['num_gauss']):
            c = np.random.uniform(0, mesh_scale, dim).astype('f')  # Scale the centers
            # s = np.random.uniform(0.1, 0.5, dim).astype('f')
            if opt['anis_gauss']:
                s = np.random.uniform(0.1, 0.5, dim).astype('f')
            else:
                s_iso = np.random.uniform(0.1, 0.5, 1).astype('f')
                s = np.repeat(s_iso, dim)

            c_list.append(c)
            s_list.append(s)
    elif opt['data_type'] == 'randg_mix':
        mesh_scale = opt.get('mesh_scale', 1.0)  # Default to 1.0 if not specified
        num_gauss = num_gauss if num_gauss is not None else opt['num_gauss']
        for j in range(num_gauss):
            c = np.random.uniform(0, mesh_scale, dim).astype('f')
            if opt['anis_gauss']:
                s = np.random.uniform(0.1, 0.5, dim).astype('f')
            else:
                if dim == 2:
                    s_iso = np.random.uniform(0.1, 0.5, 1).astype('f')
                elif dim == 3:
                    s_iso = np.random.uniform(0.05, 0.2, 1).astype('f')
                s = np.repeat(s_iso, dim)

            c_list.append(c)
            s_list.append(s)

    elif opt['data_type'] == 'structured':
        if dim == 1:  # 9 interrior points in 0.1-0.9 grid and iterate over them
            x_coord = (idx + 1) / (num_data + 1)
            c1 = np.array([x_coord]).astype('f')  # float to match torch precison
            s1 = np.array([opt['scale']]).astype('f')
            c_list.append(c1)
            s_list.append(s1)
            if opt['num_gauss'] == 2:
                c2 = np.array([0.5])
                s2 = np.array([opt['scale']])
                c_list.append(c2)
                s_list.append(s2)

        elif dim == 2:  # 25 interrior points in 0.1-0.9 grid and iterate over them plus a fixed central Gaussian
            x_coord1 = idx % 5 * 0.2 + 0.1
            y_coord1 = idx // 5 * 0.2 + 0.1
            c1 = np.array([x_coord1, y_coord1])
            s1 = np.array([opt['scale'], opt['scale']])
            c_list.append(c1)
            s_list.append(s1)
            if opt['num_gauss'] == 2:
                c2 = np.array([0.5, 0.5])
                s2 = np.array([opt['scale'], opt['scale']])
                c_list.append(c2)
                s_list.append(s2)

    pde_params = {'centers': c_list, 'scales': s_list}
    pde_params['scale_list'] = s_list
    if opt['data_type'] not in ['randg']:
        if dim == 1:
            pde_params['scale_value'] = s_list[0]  # just for naming
        elif dim == 2:
            pde_params['scale_value'] = s_list[0][0]  # just for naming

    return pde_params


def create_expansion_params(opt: dict, dim: int, idx: int, num_data: int, num_gauss: int = None) -> dict:
    #print(idx)
    #input('Press Enter to continue...')
    if opt['mesh_geometry'] in ['cylinder_100', 'cylinder_010', 'cylinder_050', 'cylinder_100_025', 'cylinder_015', 'cylinder_100_050']:
        filename_p = f"series_approximation/navier_stokes_series_coeffs_cylinder_8terms/p_expansion{idx}.json"
        filename_u = f"series_approximation/navier_stokes_series_coeffs_cylinder_8terms/u_expansion{idx}.json"
    else:
        NotImplementedError(f"Mesh geometry {opt['mesh_geometry']} not implemented.")


    pde_params = {'filename_u': filename_u, 'filename_p': filename_p}

    return pde_params