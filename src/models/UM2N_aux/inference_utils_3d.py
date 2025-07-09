"""MIT License

Copyright (c) 2024 UM2N Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# This file is based on code adapted from https://github.com/mesh-adaptation/UM2N/tree/main
# The original code is licensed under the MIT License (see above).

import firedrake as fd
import numpy as np
import torch
from firedrake.cython.dmcommon import facet_closure_nodes

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def get_conv_feat_3d(mesh, monitor_val, fix_reso_x=20, fix_reso_y=20, fix_reso_z=20):
    """
    Generate 3D convolution features by sampling the monitor function on a regular grid.
    
    Args:
        mesh: Firedrake mesh object
        monitor_val: Monitor function to sample
        fix_reso_x/y/z: Resolution of sampling grid in each dimension
    """
    coords = mesh.coordinates.dat.data_ro
    x_start, y_start, z_start = np.min(coords, axis=0)
    x_end, y_end, z_end = np.max(coords, axis=0)
    
    # Sample at fixed grid for 3D
    conv_x_fix = np.linspace(x_start, x_end, fix_reso_x)
    conv_y_fix = np.linspace(y_start, y_end, fix_reso_y)
    conv_z_fix = np.linspace(z_start, z_end, fix_reso_z)
    
    conv_monitor_val_fix = np.zeros((1, len(conv_x_fix), len(conv_y_fix), len(conv_z_fix)))
    for i in range(len(conv_x_fix)):
        for j in range(len(conv_y_fix)):
            for k in range(len(conv_z_fix)):
                try:
                    conv_monitor_val_fix[:, i, j, k] = monitor_val.at(
                        [conv_x_fix[i], conv_y_fix[j], conv_z_fix[k]], tolerance=1e-3
                    )
                except fd.function.PointNotInDomainError:
                    conv_monitor_val_fix[:, i, j, k] = 0.0
    
    return np.concatenate([conv_monitor_val_fix], axis=0)

def find_edges_3d(mesh, function_space):
    """
    Find edges in a 3D mesh.
    """
    mesh_node_count = mesh.coordinates.dat.data_ro.shape[0]
    cell_node_list = function_space.cell_node_list
    
    edge_list = []
    for cell in cell_node_list:
        # For each tetrahedron, add all 6 edges
        edges = [
            (cell[0], cell[1]), (cell[0], cell[2]), (cell[0], cell[3]),
            (cell[1], cell[2]), (cell[1], cell[3]), (cell[2], cell[3])
        ]
        edge_list.extend(edges)
    
    # Remove duplicates and convert to numpy array
    edge_list = list(set(tuple(sorted(edge)) for edge in edge_list))
    edge_index = np.array(edge_list).T
    
    # Add reverse edges for undirected graph
    edge_index_reverse = np.flip(edge_index, axis=0)
    edge_index = np.concatenate([edge_index, edge_index_reverse], axis=1)
    
    return torch.tensor(edge_index)

def find_bd_3d(mesh, function_space):
    """
    Find boundary nodes in a 3D mesh.
    """
    mesh_node_count = mesh.coordinates.dat.data_ro.shape[0]
    boundary_facets = mesh.exterior_facets
    boundary_nodes = facet_closure_nodes(mesh._topology_dm, 2, boundary_facets.facets)
    boundary_nodes = np.array(boundary_nodes, dtype=np.int32)
    
    # Create boundary mask
    bd_mask = np.zeros(mesh_node_count)
    bd_mask[boundary_nodes] = 1
    
    return bd_mask

class InputPack3D:
    """
    Data structure for 3D mesh input features.
    """
    def __init__(
        self,
        coord,
        monitor_val,
        edge_index,
        bd_mask,
        conv_feat,
        stack_boundary=True,
    ) -> None:
        self.coord = torch.tensor(coord).float().to(device)
        self.conv_feat = torch.tensor(conv_feat).float().to(device)
        
        # Normalize monitor values
        min_val = torch.min(monitor_val, dim=0).values
        max_val = torch.max(monitor_val, dim=0).values
        max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
        monitor_val = monitor_val / max_abs_val
        
        self.mesh_feat = torch.concat(
            [torch.from_numpy(coord), monitor_val.clone().detach()], 
            dim=1
        ).float().to(device)
        
        self.edge_index = edge_index.to(torch.int64).to(device)
        self.bd_mask = torch.from_numpy(bd_mask).reshape(-1, 1).to(device)
        self.node_num = self.coord.shape[0]
        
        if stack_boundary:
            # Stack boundary mask 6 times for consistency with 2D version
            self.x = torch.concat(
                [self.coord] + [self.bd_mask] * 6,
                dim=1
            ).to(device)
        else:
            self.x = torch.concat([self.coord, self.bd_mask], dim=1).to(device)
    
    def __repr__(self):
        return f"InputPack3D(nodes={self.node_num}, features={self.x.shape[1]})"
