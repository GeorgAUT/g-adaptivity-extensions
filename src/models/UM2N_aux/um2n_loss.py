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


# Below we are implementing the loss function
def UM2N_loss(x_phys, loss_func, data, use_jacob=False,
    use_inversion_loss=False,
    use_inversion_diff_loss=False,
    use_area_loss=False,
    weight_deform_loss=1.0,
    weight_area_loss=1.0,
    weight_chamfer_loss=0.0,
    scaler=100):
    # Auxiliary parameters settings taken from UM2N
    bs = 1 # Batch size parameter
    loss = 0
    inversion_loss = 0
    deform_loss = 0
    inversion_diff_loss = 0 # Note this appears not to be used in original UM2N code
    area_loss = 0
    chamfer_loss = 0

    deform_loss = 1000 * (
        loss_func(x_phys, data.x_ma)
    )

    # Extract the list of cells:
    face = torch.from_numpy(data.coarse_mesh[0].coordinates.function_space().cell_node_list).to(torch.long).T

    # Inversion loss
    if use_inversion_loss:
        inversion_loss = get_inversion_loss(x_phys, data.x_ma, face, batch_size=1, scaler=scaler)

    if use_area_loss:
        area_loss = get_area_loss(x_phys, data.x_ma, face, bs, scaler)

    chamfer_loss = 100 * chamfer_distance(x_phys.unsqueeze(0), data.x_ma.unsqueeze(0))[0]

    loss = (
            weight_deform_loss * deform_loss
            + inversion_loss
            + inversion_diff_loss
            + weight_area_loss * area_loss
            + weight_chamfer_loss * chamfer_loss
    )
    return loss

def UM2N_loss_xin(x_phys, loss_func, data, use_jacob=False,
    use_inversion_loss=False,
    use_inversion_diff_loss=False,
    use_area_loss=False,
    weight_deform_loss=1.0,
    weight_area_loss=1.0,
    weight_chamfer_loss=0.0,
    scaler=100):
    # Auxiliary parameters settings taken from UM2N
    bs = 1 # Batch size parameter
    loss = 0
    inversion_loss = 0
    deform_loss = 0
    inversion_diff_loss = 0 # Note this appears not to be used in original UM2N code
    area_loss = 0
    chamfer_loss = 0

    deform_loss = 1000 * (
        loss_func(x_phys, data.x_in)
    )

    # Extract the list of cells:
    face = torch.from_numpy(data.coarse_mesh[0].coordinates.function_space().cell_node_list).to(torch.long).T

    # Inversion loss
    if use_inversion_loss:
        inversion_loss = get_inversion_loss(x_phys, data.x_in, face, batch_size=1, scaler=scaler)

    if use_area_loss:
        area_loss = get_area_loss(x_phys, data.x_in, face, bs, scaler)

    chamfer_loss = 100 * chamfer_distance(x_phys.unsqueeze(0), data.x_in.unsqueeze(0))[0]

    loss = (
            weight_deform_loss * deform_loss
            + inversion_loss
            + inversion_diff_loss
            + weight_area_loss * area_loss
            + weight_chamfer_loss * chamfer_loss
    )
    return loss


def get_inversion_loss(
    out_coord, in_coord, face, batch_size, scheme="relu", scaler=100
):
    """
    Calculates the inversion loss for a batch of meshes.
    Args:
        out_coord (torch.Tensor): The output coordinates.
        in_coord (torch.Tensor): The input coordinates.
        face (torch.Tensor): The face tensor.
        batch_size (int): The batch size.
        alpha (float): The loss weight.
    """
    loss = None
    out_area = get_face_area(out_coord, face)
    in_area = get_face_area(in_coord, face)
    # restore the sign of the area, ans scale it
    out_area = torch.sign(in_area) * out_area
    # hard penalty, use hard condition to penalize the negative area
    if scheme == "hard":
        # mask for negative area
        neg_mask = out_area < 0
        neg_area = out_area[neg_mask]
        tar_area = in_area[neg_mask]
        # loss should be positive, so we are using -1 here.
        loss = -1 * ((neg_area / torch.abs(tar_area)).sum()) / batch_size
    # soft penalty, peanlize the negative area harder than the positive area
    elif scheme == "relu":
        loss = torch.nn.ReLU()(-1 * (out_area / torch.abs(in_area))).sum() / batch_size
    elif scheme == "log":
        epsilon = 1e-8
        loss = (
            -1 * torch.log(-1 * (out_area / torch.abs(in_area))).sum() + epsilon
        ) / batch_size
    return scaler * loss

def get_area_loss(out_coord, tar_coord, face, batch_size, scaler=100):
    out_area = get_face_area(out_coord, face)
    tar_area = get_face_area(tar_coord, face)
    # restore the sign of the area, ans scale it
    out_area = scaler * torch.sign(tar_area) * out_area
    tar_area = scaler * torch.sign(tar_area) * tar_area
    # mask for negative area
    area_diff = torch.abs(tar_area - out_area)
    # area_diff = tar_area - out_area + 100
    # loss should be positive, so we are using -1 here.
    loss = (area_diff.sum()) / batch_size
    return loss


