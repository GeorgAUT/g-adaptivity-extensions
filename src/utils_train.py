import torch


def get_face_area(coord, face):
    """
    Calculates the area of a face. using formula:
        area = 0.5 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    Args:
        coord (torch.Tensor): The coordinates.
        face (torch.Tensor): The face tensor.
    """
    x = coord[:, 0][face]
    y = coord[:, 1][face]

    area = 0.5 * (
        x[0, :] * (y[1, :] - y[2, :])
        + x[1, :] * (y[2, :] - y[0, :])
        + x[2, :] * (y[0, :] - y[1, :])
    )
    return area


def get_tetrahedron_volume(coord, face):
    """
    Calculates the signed volume of tetrahedra formed by each face and the origin.
    Uses the formula:
        volume = (1/6) * |det([v1, v2, v3])|
    where v1, v2, v3 are the edge vectors from the origin to the three vertices.

    Args:
        coord (torch.Tensor): The coordinates (N, 3).
        face (torch.Tensor): The face tensor (3, F), containing indices of the vertices.

    Returns:
        torch.Tensor: The volume of each tetrahedron.
    """
    x = coord[:, 0][face]
    y = coord[:, 1][face]
    z = coord[:, 2][face]

    # Construct vectors for the three edges of the triangle
    v1 = torch.stack([x[0], y[0], z[0]], dim=0)
    v2 = torch.stack([x[1], y[1], z[1]], dim=0)
    v3 = torch.stack([x[2], y[2], z[2]], dim=0)

    # Compute the determinant using cross product and dot product
    cross_prod = torch.cross(v2 - v1, v3 - v1, dim=0)  # (3, F)
    volume = torch.abs(torch.einsum('ij,ij->j', v1, cross_prod)) / 6.0  # (F,)

    return volume


def equidistribution_loss(x_phys, data):
    # Auxiliary parameters settings taken from UM2N
    face = torch.from_numpy(data.coarse_mesh[0].coordinates.function_space().cell_node_list).to(torch.long).T
    if x_phys.shape[1] == 2:
        out_area = get_face_area(x_phys, face)
    elif x_phys.shape[1] == 3:
        out_area = get_tetrahedron_volume(x_phys, face)
    face_Hessians = torch.mean(data.Hessian_Frob_u_tensor[face], dim=0)

    monitor_val = 1.0 + 5.0 * face_Hessians/torch.max(face_Hessians)

    equi_dist_tensor = torch.abs(out_area * monitor_val)
    loss_fn = torch.nn.MSELoss()
    mean_equi_dist_tensor = torch.mean(equi_dist_tensor)


    loss = loss_fn(equi_dist_tensor, torch.ones(equi_dist_tensor.shape) * mean_equi_dist_tensor)
    return loss