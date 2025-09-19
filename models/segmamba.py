# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import torch.nn as nn
import torch, einops
# from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

# import numpy as np
# from timm.models.layers import trunc_normal_
import math


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


# class Point:
#     def __init__(self, x=0, y=0):
#         self.x = x  # X坐标
#         self.y = y  # Y坐标
#
#
# class Hilbert:
#     def __init__(self):
#         self.hilbert_maps = {}
#         for n in [64, 32, 16, 8, 4]:
#             self.hilbert_maps[n] = self.precompute_hilbert_map(n)
#
#     def precompute_hilbert_map(self, n):
#         hilbert_map = []
#         for d in range(n * n):
#             pt = Point()
#             self.d2xy(n, d, pt)
#             hilbert_map.append([pt.x, pt.y])
#         return torch.tensor(hilbert_map, dtype=torch.long)  # 存储为张量
#
#     def rot(self, n, pt, rx, ry):
#         if ry == 0:
#             if rx == 1:
#                 pt.x = n - 1 - pt.x
#                 pt.y = n - 1 - pt.y
#
#             # Swap x and y
#             pt.x, pt.y = pt.y, pt.x
#
#     # Hilbert代码到XY坐标
#     def d2xy(self, n, d, pt):
#         pt.x, pt.y = 0, 0
#         t = d
#         s = 1
#         while s < n:
#             rx = 1 & (t // 2)
#             ry = 1 & (t ^ rx)
#             self.rot(s, pt, rx, ry)
#             pt.x += s * rx
#             pt.y += s * ry
#             t //= 4
#             s *= 2
#
#     # XY坐标到Hilbert代码转换
#     def xy2d(self, n, pt):
#         d = 0
#         s = n // 2
#         while s > 0:
#             rx = 1 if (pt.x & s) > 0 else 0
#             ry = 1 if (pt.y & s) > 0 else 0
#             d += s * s * ((3 * rx) ^ ry)
#             self.rot(s, pt, rx, ry)
#             s //= 2
#         return d


class MooreCurve:
    def __init__(self):
        self.mooreCurveMaps = {}
        # 预计算常见尺寸的映射表
        for side_length in [128, 64, 32, 16, 8, 4]:
            self.mooreCurveMaps[side_length] = self.precompute_moore_curve_map(side_length)

    def precompute_moore_curve_map(self, side_length):
        """预计算指定边长的摩尔曲线映射表"""
        # 计算阶数（根据边长推导）
        n = int(math.log2(side_length))

        # 生成坐标序列
        points = self.moore_curve_order_to_coords(n)

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)

        curve_map = []
        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            curve_map.append((y - min_y, x - min_x))

        return torch.tensor(curve_map, dtype=torch.long)

    def generate_moore_curve_string(self, n):
        """生成n阶L-system字符串 """
        axiom = 'LFL+F+LFL'
        rules = {
            'L': '-RF+LFL+FR-',
            'R': '+LF-RFR-FL+',
        }
        current = axiom
        for _ in range(n - 1):  # 替换次数为n-1次
            new_str = []
            for char in current:
                new_str.append(rules.get(char, char))  # 应用规则或保留原字符
            current = ''.join(new_str)
        return current

    def moore_curve_order_to_coords(self, n):
        """解析L-system字符串生成坐标序列"""
        string = self.generate_moore_curve_string(n)

        # 初始化状态
        x, y = 0, 0
        direction = 0  # 0:东 1:北 2:西 3:南
        points = [(x, y)]  # 包含起点

        # 解析指令
        for char in string:
            if char == 'F':
                # 根据当前方向移动
                if direction == 0:
                    x += 1
                elif direction == 1:
                    y += 1
                elif direction == 2:
                    x -= 1
                elif direction == 3:
                    y -= 1
                points.append((x, y))
            elif char == '+':
                direction = (direction - 1) % 4  # 右转
            elif char == '-':
                direction = (direction + 1) % 4  # 左转

        return points

    @staticmethod
    def coords_to_2d_array(points):
        """将坐标序列转换为二维数组（辅助函数）"""
        if not points:
            return []

        # 计算坐标范围
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # 初始化数组
        rows = max_y - min_y + 1
        cols = max_x - min_x + 1
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        # 填充遍历顺序
        for step, (x, y) in enumerate(points):
            grid[y - min_y][x - min_x] = step + 1

        return grid


class DiagonalMap:
    def __init__(self):
        self.diagonalMaps = {}
        # 预计算常见尺寸的映射表
        for side_length in [128, 64, 32, 16, 8, 4]:
            self.diagonalMaps[side_length] = self.precompute_diagonal_order(side_length)

    def precompute_diagonal_order(self, side_length):
        """预计算坐标映射关系，返回一维坐标到二维坐标的列表"""
        H = side_length
        W = side_length
        coord_map = []
        for s in range(H + W - 1):
            start_row = max(0, s - (W - 1))
            end_row = min(H - 1, s)
            for row in range(end_row, start_row - 1, -1):  # 逆序保证右上到左下
                coord = (row, s - row)  # 存储二维坐标
                coord_map.append(coord)
        return torch.tensor(coord_map, dtype=torch.long)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, mooreMaps=None, diagonalMaps=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mooreMaps = mooreMaps
        self.diagonalMaps = diagonalMaps
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    # def hilbertFlat(self, tensor, hilbertMap):
    #     # 仅适用于正方形图片
    #     # 获取输入张量的形状
    #     batch_size, channel, h, w, d = tensor.shape
    #     n = h * w * d
    #     # 初始化结果张量
    #     flat = torch.zeros((batch_size, channel, n), dtype=tensor.dtype, device=tensor.device)
    #     frameSize = h * w  # 每一帧的大小
    #     for b in range(batch_size):
    #         for s in range(channel):
    #             for i in range(n):
    #                 # 填满一帧
    #                 f = i // frameSize
    #                 # 按照hilBert曲线的顺序填充张量
    #                 flat[b, s, i] = tensor[b, s, hilbertMap[i%frameSize][0], hilbertMap[i%frameSize][1], f]
    #
    #     return flat

    # def hilbertReshape(self, flatTensor, hilbertMap):
    #     batch_size, channel, hilbertSeq = flatTensor.shape
    #     cubeSize = int(math.ceil(hilbertSeq ** (1 / 3)))
    #     # 初始化结果张量
    #     reshapeTensor = torch.zeros((batch_size, channel, cubeSize, cubeSize, cubeSize), dtype=flatTensor.dtype,
    #                                 device=flatTensor.device)
    #     frameSize = cubeSize * cubeSize  # 每一帧的大小
    #     for b in range(batch_size):
    #         for s in range(channel):
    #             for i in range(hilbertSeq):
    #                 # 填满一帧
    #                 f = i // frameSize
    #                 # 按照hilBert曲线的顺序填充张量
    #                 reshapeTensor[b, s, hilbertMap[i%frameSize][0], hilbertMap[i%frameSize][1], f] = flatTensor[b, s, i]
    #
    #     return reshapeTensor
    def DiagonalFlat(self, tensor, diagonalMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = diagonalMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def DiagonalReshape(self, flatTensor, diagonalMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = diagonalMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = diagonalMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor

    def mooreFlat(self, tensor, mooreMap):
        B, C, h, w, d = tensor.shape
        device = tensor.device
        coord_map = mooreMap.to(device)

        total_elements = h * w * d
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        i_in_frame = i_all % frame_size
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')

        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        flat = tensor[:, :, x_idx, y_idx, f_idx]
        return flat

    def mooreReshape(self, flatTensor, mooreMap):
        B, C, total_elements = flatTensor.shape
        device = flatTensor.device
        hw = mooreMap.size(0)
        h = int(math.sqrt(hw))
        w = h
        d = total_elements // hw

        reshapeTensor = torch.zeros((B, C, h, w, d), dtype=flatTensor.dtype, device=device)
        i_all = torch.arange(total_elements, device=device)
        frame_size = h * w
        # f_idx = i_all // frame_size
        f_idx = torch.div(i_all, frame_size, rounding_mode='trunc')
        i_in_frame = i_all % frame_size

        coord_map = mooreMap.to(device)
        x_y = coord_map[i_in_frame]
        x_idx, y_idx = x_y[:, 0], x_y[:, 1]

        reshapeTensor[:, :, x_idx, y_idx, f_idx] = flatTensor
        return reshapeTensor

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        img_dims_dhw = x.permute(0, 1, 4, 3, 2).shape[2:]

        # 后三维展平为一维进行扫描
        # 横向
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # 恢复为原来的形状
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # 竖向
        # x_flat_vertical = x.permute(0, 1, 4, 3, 2).reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat_vertical)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims_dhw).permute(0, 1, 3, 4, 2)

        # 斜向
        # frameSize = x.shape[-2]
        # diagonalMap = self.diagonalMaps[frameSize]
        # x_dgflat = self.DiagonalFlat(x, diagonalMap).transpose(-1, -2)
        # x_norm = self.norm(x_dgflat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = self.DiagonalReshape(x_mamba.transpose(-1, -2), diagonalMap)

        # MRscan
        # frameSize = x.shape[-2]
        # mooreMap = self.mooreMaps[frameSize]
        # x_MRflat = self.mooreFlat(x, mooreMap).transpose(-1, -2)
        # x_norm = self.norm(x_MRflat)
        # x_mamba = self.mamba(x_norm)
        # # 恢复为原来的形状
        # out = self.mooreReshape(x_mamba.transpose(-1, -2), mooreMap)

        out = out + x_skip

        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], mooreMaps=None,
                 diagonalMaps=None):
        super().__init__()

        self.mooreMaps = mooreMaps
        self.diagonalMaps = diagonalMaps
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i], mooreMaps=mooreMaps, diagonalMaps=diagonalMaps)
                  for j in
                  range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SegMamba(nn.Module):
    def __init__(
            self,
            in_chans=3,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        # 初始化 moore的mooreMap
        mooreMaps = MooreCurve().mooreCurveMaps
        diagonalMaps = DiagonalMap().diagonalMaps

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                mooreMaps=mooreMaps,
                                diagonalMaps=diagonalMaps
                                )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=16, out_channels=self.out_chans)
        # self.out = UnetOutBlock(spatial_dims=2, in_channels=16, out_channels=self.out_chans)


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)

        x2 = outs[0]
        enc2 = self.encoder2(x2)

        x3 = outs[1]
        enc3 = self.encoder3(x3)
        # ex3 = enc3

        x4 = outs[2]
        enc4 = self.encoder4(x4)

        enc_hidden = self.encoder5(outs[3])
        # ex5 = enc_hidden

        dec3 = self.decoder5(enc_hidden, enc4)

        dec2 = self.decoder4(dec3, enc3)

        dec1 = self.decoder3(dec2, enc2)

        dec0 = self.decoder2(dec1, enc1)

        # origin
        out = self.decoder1(dec0)

        # # 进行可视化
        # # (1,c,h,w,f)
        # visualizer = ImageVisualizer()
        #
        # #visualizer.show_image(dx1[0, 0, :, :, 0], cmap='jet', save_path='/media/gx/code/data/cn24/program/SimLVSeg/visualization/image/decoder1.png')
        #
        # vis_img = ex3
        # image_list = []
        # for c in range(4):
        #     c_img = vis_img[0, c, :, :, 0]
        #     image_list.append(c_img)
        # # visualizer.show_images(image_list, cmap='jet', save_path='/media/gx/code/data/cn24/program/SimLVSeg/visualization/image/att_decoder2.png')
        # visualizer.show_images(image_list, cmap='gray', save_path='/workdir1/cn24/program/SimLVSeg/visualization/image/encoder3.png')

        return self.out(out)

# 查看模型
# if __name__ == "__main__":
#
#     # 加载模型权重
#     # 加载 checkpoint 文件
#     checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_249/checkpoints/epoch=55-step=208878.ckpt'
#     # checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=True)
#     checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
#
#     # 获取 state_dict
#     state_dict = checkpoint['state_dict']
#
#     # 移除 'model.' 前缀
#     new_state_dict = {}
#     for key in state_dict:
#         new_key = key.replace("model.", "")  # 移除 'model.' 前缀
#         new_state_dict[new_key] = state_dict[key]
#
#     # 初始化模型
#     # model = UNet3D().cuda(1)  # 或者使用 UNet3DSmall()
#     # model = OnlyUKAN3D().cuda(1)  # 或者使用 UNet3DSmall()
#
#     # 加载移除前缀后的 state_dict
#     model = SegMamba(in_chans=3,
#                      out_chans=1,
#                      depths=[2, 2, 2, 2],
#                      feat_size=[16, 32, 64, 128])
#     model.load_state_dict(new_state_dict)
#
#     # 将模型加载到 GPU 0
#     model = model.cuda(0)
#
#     # 伪造输入数据：假设输入形状为 (batch_size=1, channels=3, depth=128, height=128, width=128)
#     # input = torch.randn(1, 3, 112, 112, 32).cuda(1)
#
#     # 使用真实数据
#     from utils.img2tensor import video_to_tensor
#
#     video_tensor = video_to_tensor(
#         r'/workdir1/echo_dataset/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi')
#     target_shape = (128, 128, 32)
#     input = video_tensor[:, :, :, :, 0:32].cuda(0)
#     # 计算每个维度需要填充的像素数
#     pad_h = (8, 8)  # 112 -> 128
#     pad_w = (8, 8)  # 112 -> 128
#
#     # 使用零填充到目标形状
#     input = F.pad(input, (0, 0, pad_w[0], pad_w[1], pad_h[0], pad_h[1]), "constant", 0)
#
#     print(input.shape)
#
#     input_tensor = input.float()
#     # 前向传播
#     output = model(input_tensor)
#
#     from thop import profile
#
#     flops, params = profile(model, inputs=(input,))
#     print('Flops: ', flops, ', Params: ', params)
#     print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
#
#     # 打印输出形状
#     # print(f"输出形状: {output.shape}")


# IOU可视化
if __name__ == "__main__":

    # 测试真实数据
    ####################################################################################################################

    # step1 加载模型权重
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_286/checkpoints/epoch=35-step=33086.ckpt'
    # 儿科 ped
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_201/checkpoints/epoch=36-step=45916.ckpt'
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_239/checkpoints/epoch=35-step=44055.ckpt'
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_236/checkpoints/epoch=45-step=57085.ckpt'
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_245/checkpoints/epoch=36-step=11315.ckpt'
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_292/checkpoints/epoch=49-step=61429.ckpt'
    # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_302/checkpoints/epoch=30-step=18910.ckpt'
    # echo-dynamic
    checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_253/checkpoints/epoch=42-step=157592.ckpt'

    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

    # 获取 state_dict
    state_dict = checkpoint['state_dict']

    # 移除 'model.' 前缀
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("model.", "")  # 移除 'model.' 前缀
        new_state_dict[new_key] = state_dict[key]

    # step2 初始化模型并加载权重
    model = SegMamba(in_chans=3,
                     out_chans=1,
                     depths=[2, 2, 2, 2],
                     feat_size=[16, 32, 64, 128])
    model.load_state_dict(new_state_dict)
    model.eval()

    # 将模型加载到 GPU 0
    model = model.cuda(0)
    # print(model)


    # step3 加载真实数据
    import numpy as np
    from utils.img2tensor import video_to_tensor, video_resize_to_tensor, video_pad_to_tensor
    from utils.result_vis import visualize_segmentation
    from utils.csv2mask import create_mask_from_csv_dynamic, create_mask_from_csv_pediatric

    # video_name = r'0X112EF236E1F676E4.avi'
    # video_name = r'0X129133A90A61A59D.avi'
    # video_name = r'0X10A28877E97DF540.avi'
    # video_name = r'0X1C48B213563D806E.avi'
    # video_name = r'0X1AB987597AF39E3B.avi'
    video_name = r'0X3154AED0655FDFA.avi'

    # video_name = r'CR32a969d-CR32a9906-000084.avi'

    video_path = r'/workdir1/echo_dataset/EchoNet-Dynamic/Videos/' + video_name
    # video_path = r'/workdir1/cn24/data/pediatric_echo/A4C/Videos/' + video_name
    # save_name = "visualize_video/v201_M_F.avi"
    # save_name = "visualize_video/v239___F.avi"
    # save_name = "visualize_video/v236_M__.avi"
    # save_name = "visualize_video/v245____.avi"
    # save_name = "visualize_video/v292____.avi"
    # save_name = "visualize_video/v302____.avi"
    save_name = "visualize_video/origin_dynamic_hot.avi"
    ed_frame = 112
    es_frame = -1

    # get Echonet-Dynamic pt
    ground_truth = create_mask_from_csv_dynamic(
        r'/workdir1/echo_dataset/EchoNet-Dynamic/VolumeTracings.csv',
        video_name,
        ed_frame,
        128,
        128)

    # get Echonet-Pediatric pt
    # ground_truth = create_mask_from_csv_pediatric(
    #     r'/workdir1/cn24/data/pediatric_echo/A4C/VolumeTracings.csv',
    #     video_name,
    #     ed_frame,
    #     128,
    #     128)

    ground_truth = np.tile(ground_truth, (32, 1, 1))

    # CAMUS
    # ground_truth_path = r'/workdir1/cn24/data/CAMUS/patient0043_a4c_gt.npy'
    # ground_truth = np.load(ground_truth_path)
    #
    # ground_truth = np.tile(ground_truth, (32, 1, 1))
    #
    # video_name = r'patient0043_a4c_seq'
    # video_path = r'/workdir1/cn24/data/CAMUS/' + video_name

    video_tensor = video_pad_to_tensor(video_path)
    video_tensor = video_tensor[..., ed_frame - 1:]

    # 获取视频的帧数
    num_frames = video_tensor.shape[-1]

    # 如果视频帧数小于 32，使用镜像复制的方式填充
    if num_frames < 32:
        # 使用最后一帧进行镜像复制填充
        repeat_frames = 32 - num_frames
        video_tensor = torch.cat([video_tensor, video_tensor[:, :, :, :, -1:].repeat(1, 1, 1, 1, repeat_frames)],
                                 dim=-1)

    # 确保 input 取到前 16 帧
    input = video_tensor[:, :, :, :, 0:32].cuda(0)

    # step4 前向传播
    output = model(input)

    # 可视化第一帧
    # gradcam(model, input, target_class=0)  # 选择类别，通常是背景或者目标类

    # 统计参数量
    # from thop import profile
    #
    # flops, params = profile(model, inputs=(input,))
    # print('Flops: ', flops, ', Params: ', params)
    # print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))

    # 打印输出形状
    print(f"输出形状: {output.shape}")

    # 可视化并保存
    visualize_segmentation(video_path, output, ground_truth, save_name, 0, 0, [122, 200, 121], [134, 90, 190],[255, 223, 128])
    # visualize_segmentation_jet(video_path, output, grount_truth, "375.avi")

    # 保存输出的mask视频
    # tensor_to_video_grayscale(output, 'output.avi')

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from utils.camutils import GradCAM, show_cam_on_image
#
#     # 加载模型权重
#     # checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_302/checkpoints/epoch=30-step=18910.ckpt'
#     checkpoint_path = r'/workdir1/cn24/program/SimLVSeg/lightning_logs/version_253/checkpoints/epoch=42-step=157592.ckpt'
#     checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
#
#     # 获取 state_dict
#     state_dict = checkpoint['state_dict']
#
#     # 移除 'model.' 前缀
#     new_state_dict = {}
#     for key in state_dict:
#         new_key = key.replace("model.", "")  # 移除 'model.' 前缀
#         new_state_dict[new_key] = state_dict[key]
#
#     # step2 初始化模型并加载权重
#     model = SegMamba(in_chans=3,
#                      out_chans=1,
#                      depths=[2, 2, 2, 2],
#                      feat_size=[16, 32, 64, 128])
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     # 将模型加载到 GPU 0
#     model = model.cuda(0)
#
#
#     # step3 加载真实数据
#     import numpy as np
#     from utils.img2tensor import video_pad_to_tensor
#
#     # video_name = r'0X112EF236E1F676E4.avi'
#     # video_name = r'0X129133A90A61A59D.avi'
#     # video_name = r'0X10A28877E97DF540.avi'
#     # video_name = r'0X1C48B213563D806E.avi'
#     # video_name = r'0X1AB987597AF39E3B.avi'
#     video_name = r'0X3154AED0655FDFA.avi'
#     video_path = r'/workdir1/echo_dataset/EchoNet-Dynamic/Videos/' + video_name
#     # video_path = r'/workdir1/cn24/data/pediatric_echo/A4C/Videos/' + video_name
#     # save_name = "visualize_video/cam_out.png"
#     # save_name = "visualize_video/cam_decoder1.png"
#     save_name = "visualize_video/cam_decoder2.png"
#     ed_frame = 112
#
#     video_tensor = video_pad_to_tensor(video_path)
#     video_tensor = video_tensor[..., ed_frame - 1:]
#
#     # 获取视频的帧数
#     num_frames = video_tensor.shape[-1]
#
#     # 如果视频帧数小于 32，使用镜像复制的方式填充
#     if num_frames < 32:
#         # 使用最后一帧进行镜像复制填充
#         repeat_frames = 32 - num_frames
#         video_tensor = torch.cat([video_tensor, video_tensor[:, :, :, :, -1:].repeat(1, 1, 1, 1, repeat_frames)],
#                                  dim=-1)
#     # 确保 input 取到前 16 帧
#     input = video_tensor[:, :, :, :, 0:32].cuda(0)
#
#     # 获取模型最后一层作为目标层
#     # target_layers = [model.out]
#     # target_layers = [model.decoder1]
#     target_layers = [model.decoder2]
#
#     # 创建 GradCAM 对象
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#     # 指定目标类别
#     target_category = 0
#
#     grayscale_cam = cam(input_tensor=input, target_category=target_category)
#     print(f"grayscale_cam.shape: {grayscale_cam.shape}")
#     grayscale_cam = grayscale_cam[0, :]
#
#     # 可视化
#     video_array = video_tensor.detach().cpu().numpy().astype(np.float32) / 255.
#     # 1. 去掉 batch 维度 (0) 和 frame 维度 (-1)
#     video_squeezed = video_array[0, :, :, :, 0]  # 取第一个 batch 和第一个 frame
#     # 2. 将通道维度放到最后 -> [H, W, C]
#     video_final = np.transpose(video_squeezed, (1, 2, 0))
#     print(video_final.shape)
#     visualization = show_cam_on_image(video_final,
#                                       grayscale_cam,
#                                       use_rgb=True)
#     # 使用 matplotlib 保存图像
#     plt.imshow(visualization)
#     plt.axis('off')  # 可选：隐藏坐标轴
#     plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
#     plt.close()
#
#     print("热力图已保存")



