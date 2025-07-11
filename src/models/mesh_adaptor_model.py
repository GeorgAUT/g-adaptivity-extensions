# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code derived from:
# https://github.com/mesh-adaptation/UM2N/tree/main
# which is licensed under the MIT License (included below).
#
# -----------------------------------------------------------------------------
# The following applies to portions of this file adapted from UM2N:
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

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from firedrake import *
from firedrake.__future__ import interpolate

from models.UM2N_aux.extractor import LocalFeatExtractor, TransformerEncoder
from models.diffformer_block import Diffformer
from models.UM2N_aux.gatdeformer import DeformGAT
from models.UM2N_aux.inference_utils import InputPack, find_bd, get_conv_feat
from models.UM2N_aux.inference_utils_3d import get_conv_feat_3d


class NetDeform(torch.nn.Module):
    def __init__(self, in_dim, opt=None):
        super(NetDeform, self).__init__()
        self.opt = opt
        self.dim = len(self.opt['mesh_dims'])
        self.lin = torch.nn.Linear(in_dim, 254)
        self.lin_out1 = 254
        self.deform_out1 = 508
        self.deform_out2 = 250
        self.deform_out3 = 120
        self.deform_out4 = 20
        if self.opt['grand_diffusion']:
            self.gat_1 = Diffformer(self.lin_out1 + self.dim, self.deform_out1, heads=6, opt=opt)
            self.gat_2 = Diffformer(self.deform_out1 + 2 * self.dim, self.deform_out2, heads=6, opt=opt)
            self.gat_3 = Diffformer(self.deform_out2 + 3 * self.dim, self.deform_out3, heads=6, opt=opt)
            self.gat_4 = Diffformer(self.deform_out3 + 4 * self.dim, 20, heads=6, opt=opt)
        else:
            self.gat_1 = DeformGAT(self.lin_out1 + self.dim, self.deform_out1, heads=6)
            self.gat_2 = DeformGAT(self.deform_out1 + 2 * self.dim, self.deform_out2, heads=6)
            self.gat_3 = DeformGAT(self.deform_out2 + 3 * self.dim, self.deform_out3, heads=6)
            self.gat_4 = DeformGAT(self.deform_out3 + 4 * self.dim, 20, heads=6)

        self.layer_evolution = []

    def forward(self, data, edge_idx, bd_mask, poly_mesh):
        coords_tensor = data[:, 0:self.dim]
        lin_1 = self.lin(data)
        lin_1 = F.selu(lin_1)
        together_1 = torch.cat([coords_tensor, lin_1], dim=1)

        out_coord_1, out_feature_1 = self.gat_1(
            coords_tensor, together_1, edge_idx, bd_mask, poly_mesh
        )

        together_2 = torch.cat([out_coord_1, coords_tensor, out_feature_1], dim=1)
        out_coord_2, out_feature_2 = self.gat_2(
            out_coord_1, together_2, edge_idx, bd_mask, poly_mesh
        )

        together_3 = torch.cat(
            [out_coord_2, out_coord_1, coords_tensor, out_feature_2], dim=1
        )
        out_coord_3, out_feature_3 = self.gat_3(
            out_coord_2, together_3, edge_idx, bd_mask, poly_mesh
        )

        together_4 = torch.cat(
            [out_coord_3, out_coord_2, out_coord_1, coords_tensor, out_feature_3], dim=1
        )
        out_coord_4, out_feature_4 = self.gat_4(
            out_coord_3, together_4, edge_idx, bd_mask, poly_mesh
        )

        if self.opt and self.opt.get('show_mesh_evol_plots', False):
            self.layer_evolution.append({
                'layer0': coords_tensor.detach().clone(),
                'layer1': out_coord_1.detach().clone(),
                'layer2': out_coord_2.detach().clone(),
                'layer3': out_coord_3.detach().clone(),
                'layer4': out_coord_4.detach().clone(),
                'edge_index': edge_idx.clone()
            })

        return out_coord_4


class Mesh_Adaptor(nn.Module):
    def __init__(self, opt, gfe_in_c=3, lfe_in_c=3, deform_in_c=3):
        self.use_patches = opt.get('use_patches', False)

        self.opt = opt

        super().__init__()
        self.gfe_out_c = 16
        self.lfe_out_c = 16
        self.deformer_in_feat = deform_in_c + self.gfe_out_c + self.lfe_out_c

        self.gfe = TransformerEncoder(
            num_transformer_in=gfe_in_c, num_transformer_out=self.gfe_out_c
        )
        self.lfe = LocalFeatExtractor(num_feat=lfe_in_c, out=self.lfe_out_c)
        self.deformer = NetDeform(in_dim=self.deformer_in_feat, opt=self.opt)

    def patch_mesh_feat(self, mesh_feat, patch_indices):
        """Convert mesh features into patch-based format
        Args:
            mesh_feat: [num_nodes, feat_dim] tensor of mesh features
            patch_indices: list of lists of node indices for each patch
        Returns:
            [num_patch_nodes, feat_dim] tensor of patched features
        """
        # Just concatenate all patch features
        all_patch_indices = np.concatenate(patch_indices)
        if all_patch_indices.max() >= mesh_feat.shape[0]:
            raise ValueError(f'Patch index {all_patch_indices.max()} >= mesh_feat dim 0 ({mesh_feat.shape[0]})')
        return mesh_feat[torch.tensor(all_patch_indices, device=mesh_feat.device)]
    
    def process_batch(self, data):
        """Convert PyG data batch to InputPack format"""

        # Extract input features
        coords = data.x_in.detach().cpu().numpy()
        V = FunctionSpace(data.coarse_mesh[0], "CG", 1)
        edge_idx = data.edge_index
        bd_mask, _, _, _, _ = find_bd(data.coarse_mesh[0], V)

        monitor_val = Function(FunctionSpace(data.coarse_mesh[0], "CG", 1))
        if self.opt['new_model_monitor_type']=='UM2N':
            monitor_val = monitor_func(data.coarse_mesh[0], data.u_coarse_reg[0])
            monitor_val_data = monitor_val.dat.data[:]
        elif self.opt['new_model_monitor_type']=='Hessian_Frob_u_tensor':
            monitor_val_data = data.Hessian_Frob_u_tensor.numpy()
        else:
            raise ValueError("Invalid monitor type in new GNN model!")

        filter_monitor_val = np.minimum(1e3, monitor_val_data)
        filter_monitor_val = np.maximum(0, filter_monitor_val)
        monitor_val.dat.data[:] = filter_monitor_val / filter_monitor_val.max()
        dim = coords.shape[1]
        if dim == 3:
            conv_feat = get_conv_feat_3d(data.coarse_mesh[0], monitor_val)
        else:
            conv_feat = get_conv_feat(data.coarse_mesh[0], monitor_val)
        start_time = time.perf_counter()
        sample = InputPack(
            coord=coords,
            monitor_val=torch.tensor(monitor_val.dat.data_ro.reshape(-1, 1)),
            edge_index=edge_idx,
            bd_mask=bd_mask,
            conv_feat=conv_feat,
            stack_boundary=False,
        )

        return sample

    def forward(self, data_in, poly_mesh=False):

        data = self.process_batch(data_in)

        self.start_MLmodel = time.time()
        bd_mask = data.bd_mask
        if data.poly_mesh is not False:
            poly_mesh = True if data.poly_mesh.sum() > 0 else False
        x = data.x
        batch_size = data.conv_feat.shape[0]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
        edge_idx = data.edge_index  # [num_edges * batch_size, 2]
        feat_dim = mesh_feat.shape[-1]
        
        if self.use_patches:
            patched_feat = self.patch_mesh_feat(mesh_feat, data_in.patch_indices[0])
            global_feat = self.gfe(patched_feat)
        else:
            global_feat = self.gfe(mesh_feat.view(batch_size, -1, feat_dim))
        local_feat = self.lfe(mesh_feat, edge_idx)
        x = torch.cat([x, local_feat, global_feat], dim=1)
        x = self.deformer(x, edge_idx, bd_mask, poly_mesh)

        self.end_MLmodel = time.time()
        return x


def monitor_func(mesh, u, alpha=5.0):
    vec_space = VectorFunctionSpace(mesh, "CG", 1)
    if u.dat.data_ro.ndim == 1:
        vec_space = VectorFunctionSpace(mesh, "CG", 1)
        uh_grad = assemble(interpolate(grad(u), vec_space))
        grad_norm = Function(FunctionSpace(mesh, "CG", 1))
        grad_norm.interpolate(uh_grad[0] ** 2 + uh_grad[1] ** 2)
        return grad_norm
    else:
        grad_norm_total = Function(FunctionSpace(mesh, "CG", 1))
        for j in range(len(u)):
            grad_norm = Function(FunctionSpace(mesh, "CG", 1))
            uh_grad = assemble(interpolate(grad(u[j]), vec_space))
            grad_norm.interpolate(uh_grad[0] ** 2 + uh_grad[1] ** 2)
            grad_norm_total += grad_norm
        grad_norm_total = assemble(grad_norm_total)
        return grad_norm_total