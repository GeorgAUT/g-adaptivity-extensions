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

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import softmax

__all__ = ["Diffformer"]


class Diffformer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(Diffformer, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = False

        self.lin_l = Linear(in_channels, heads * out_channels, bias=True).float()
        self.lin_ = self.lin_l

        self.att_l = Parameter(torch.FloatTensor(1, heads, out_channels))
        self.att_r = Parameter(torch.FloatTensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.FloatTensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.negative_slope = -0.2
        self.reset_parameters()

        self.opt = kwargs['opt']
        self.layer_evolution = []

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_.weight)
        glorot(self.att_l)
        glorot(self.att_r)

    def forward(
        self,
        coords: Union[Tensor, OptPairTensor],
        features: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        bd_mask,
        poly_mesh,
    ):
        if self.opt['show_mesh_evol_plots']:
            self.layer_evolution.append(coords.detach().clone())

        self.bd_mask = bd_mask.squeeze().bool()
        self.poly_mesh = poly_mesh
        self.find_boundary(coords)
        H, C = self.heads, self.out_channels
        x_l = x_r = self.lin_l(features).view(-1, H, C)  # [num_node , heads, out_channels]

        x_coords_l = x_coords_r = coords

        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        if self.opt['grand_diffusion']:
            for i in range(self.opt['grand_diffusion_steps']):
                x_coords_l = x_coords_r = coords.unsqueeze(1)
                out_coords = self.propagate(
                    edge_index, x=(x_coords_l, x_coords_r), alpha=(0.2 * alpha_l, 0.2 * alpha_r)
                )

                out_coords = out_coords.mean(dim=1)
                coords = (1 - self.opt['grand_step_size']) * coords + self.opt['grand_step_size'] * out_coords

                self.fix_boundary(coords)
            out_coords = coords
        else:
            x_coords_l = x_coords_r = coords.unsqueeze(1)

            out_coords = self.propagate(
                edge_index, x=(x_coords_l, x_coords_r), alpha=(0.2 * alpha_l, 0.2 * alpha_r)
            )

            out_coords = out_coords.mean(dim=1)

        out_features = self.propagate(
            edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r)
        )

        out_features = out_features.mean(dim=1)
        out_features = F.selu(out_features)

        self.fix_boundary(out_coords)

        return out_coords, out_features

    def is_self_loop(self, edge_index):
        return edge_index[0] == edge_index[1]

    def message(
            self,
            x_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int],
    ) -> Tensor:
        if alpha_i is None:
            alpha = alpha_j
        else:
            alpha = (
                    alpha_j + alpha_i
            )
        alpha = F.selu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.unsqueeze(-1)


    def find_boundary(self, in_data):
        self.upper_node_idx = in_data[:, 0] == 1
        self.down_node_idx = in_data[:, 0] == 0
        self.left_node_idx = in_data[:, 1] == 0
        self.right_node_idx = in_data[:, 1] == 1

        # if self.poly_mesh:
        self.bd_pos_x = in_data[self.bd_mask, 0].clone()
        self.bd_pos_y = in_data[self.bd_mask, 1].clone()
        if in_data.shape[1]==3:
            self.bd_pos_z = in_data[self.bd_mask, 2].clone()

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1

        # if self.poly_mesh:
        in_data[self.bd_mask, 0] = self.bd_pos_x
        in_data[self.bd_mask, 1] = self.bd_pos_y
        if in_data.shape[1]==3:
            in_data[self.bd_mask, 2] = self.bd_pos_z

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )