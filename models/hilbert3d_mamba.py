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

class Hilbert3DMapper:
    def __init__(self):
        """
        初始化并预计算常见尺寸的 Hilbert 3D 映射表。
        字典的 Key 为立方体边长 (例如 4, 8, 16...)
        字典的 Value 为映射列表 [(x,y,z), (x,y,z), ...]
        """
        self.hilbert3dMaps = {}

        # 预计算常见尺寸的映射表
        # 分别对应 128*128*128, 64*64*64, 32*32*32 等立方体
        for side_length in [128, 64, 32, 16, 8, 4]:
            # 根据边长反推迭代次数: side_length = 2 ** (iterations + 1)
            iterations = int(math.log2(side_length)) - 1
            print(f"正在预计算边长 {side_length} (迭代次数 {iterations}) 的 Hilbert 映射...")
            self.hilbert3dMaps[side_length] = self._generate_1d_to_3d_mapping(iterations)

        print("所有预计算完成！\n" + "=" * 40)

    def _generate_1d_to_3d_mapping(self, iterations):
        """
        核心生成算法 (原 get_1d_to_3d_mapping)
        """

        def hilbert3d(center, size, it, v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7):
            half = size / 2.0
            x, y, z = center

            vec_s = [
                (x - half, y + half, z - half),
                (x - half, y + half, z + half),
                (x - half, y - half, z + half),
                (x - half, y - half, z - half),
                (x + half, y - half, z - half),
                (x + half, y - half, z + half),
                (x + half, y + half, z + half),
                (x + half, y + half, z - half)
            ]

            vec = [
                vec_s[v0], vec_s[v1], vec_s[v2], vec_s[v3],
                vec_s[v4], vec_s[v5], vec_s[v6], vec_s[v7]
            ]

            it -= 1
            if it >= 0:
                tmp = []
                tmp.extend(hilbert3d(vec[0], half, it, v0, v3, v4, v7, v6, v5, v2, v1))
                tmp.extend(hilbert3d(vec[1], half, it, v0, v7, v6, v1, v2, v5, v4, v3))
                tmp.extend(hilbert3d(vec[2], half, it, v0, v7, v6, v1, v2, v5, v4, v3))
                tmp.extend(hilbert3d(vec[3], half, it, v2, v3, v0, v1, v6, v7, v4, v5))
                tmp.extend(hilbert3d(vec[4], half, it, v2, v3, v0, v1, v6, v7, v4, v5))
                tmp.extend(hilbert3d(vec[5], half, it, v4, v3, v2, v5, v6, v1, v0, v7))
                tmp.extend(hilbert3d(vec[6], half, it, v4, v3, v2, v5, v6, v1, v0, v7))
                tmp.extend(hilbert3d(vec[7], half, it, v6, v5, v2, v1, v0, v3, v4, v7))
                return tmp

            return vec

        initial_size = 2 ** iterations
        raw_points = hilbert3d((0.0, 0.0, 0.0), initial_size, iterations)

        min_x = min(p[0] for p in raw_points)
        min_y = min(p[1] for p in raw_points)
        min_z = min(p[2] for p in raw_points)

        mapped_sequence = []
        for p in raw_points:
            ix = int(round(p[0] - min_x))
            iy = int(round(p[1] - min_y))
            iz = int(round(p[2] - min_z))
            mapped_sequence.append((ix, iy, iz))

        return mapped_sequence

    def get_mapping(self, side_length):
        """
        获取指定边长的 1D 到 3D 映射表。
        如果请求的尺寸未在预计算中，则会实时计算并缓存。
        """
        if side_length not in self.hilbert3dMaps:
            iterations = int(math.log2(side_length)) - 1
            self.hilbert3dMaps[side_length] = self._generate_1d_to_3d_mapping(iterations)
        return self.hilbert3dMaps[side_length]


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


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, hilbertMaps=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # 接收上层传进来的预计算 Hilbert 映射字典
        self.hilbertMaps = hilbertMaps

        # 缓存映射到 device 上的 PyTorch 1D 索引 (避免每次前向传播重复构建)
        self.forward_indices = {}
        self.inverse_indices = {}

        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def _get_indices(self, S, device):
        """
        根据立方体边长 S 获取/生成 1D 的展平和复原索引张量
        """
        if S not in self.forward_indices:
            if self.hilbertMaps is None or S not in self.hilbertMaps:
                raise ValueError(f"缺少边长为 {S} 的 Hilbert 映射数据，请在 Hilbert3DMapper 中预计算。")

            mapping = self.hilbertMaps[S]

            # 1. 构建前向索引 P (将 (x,y,z) 转换为一维线性索引)
            # PyTorch 的 view(B, C, -1) 展平顺序是 D, H, W (对应 x, y, z)
            # 线性索引公式: idx = x * S^2 + y * S + z
            P = [x * (S * S) + y * S + z for (x, y, z) in mapping]
            P_tensor = torch.tensor(P, dtype=torch.long, device=device)

            # 2. 构建反向索引 P_inv (用于 Reshape 还原)
            P_inv_tensor = torch.empty_like(P_tensor)
            P_inv_tensor[P_tensor] = torch.arange(len(P), device=device)

            self.forward_indices[S] = P_tensor
            self.inverse_indices[S] = P_inv_tensor

        return self.forward_indices[S], self.inverse_indices[S]

    def HB3D_Flat(self, x):
        """
        按照 Hilbert 曲线将 3D 张量展平为 1D 序列
        x shape: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        S = D  # 假设输入是一个立方体 D == H == W

        # 获取展平用的排列索引 P
        P, _ = self._get_indices(S, x.device)

        # 先按常规 C-contiguous 展平前三个空间维度: shape -> (B, C, L)
        x_flat = x.view(B, C, -1)

        # 按照 Hilbert 顺序重新排列
        x_hilbert = x_flat[:, :, P]

        return x_hilbert

    def HB3D_Reshape(self, x_flat, S):
        """
        将 Mamba 处理后的 Hilbert 一维序列还原回 3D 张量
        x_flat shape: (B, C, L)
        S: 立方体边长
        """
        B, C, L = x_flat.shape

        # 获取复原用的反向排列索引 P_inv
        _, P_inv = self._get_indices(S, x_flat.device)

        # 将 Hilbert 顺序还原为常规 C-contiguous 的一维顺序
        x_restored_flat = x_flat[:, :, P_inv]

        # reshape 回 3D 结构
        x_restored = x_restored_flat.view(B, C, S, S, S)

        return x_restored

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim

        # 假设空间维度是规则的立方体，提取边长 S
        S = x.shape[-1]

        # --- 第一部分：标准的横向扫描 (如果需要保留的话) ---
        # n_tokens = x.shape[2:].numel()
        # img_dims = x.shape[2:]
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat)
        # x_mamba = self.mamba(x_norm)
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        # --- 第二部分：HB3D scan (Hilbert 3D) ---
        # 1. 映射到 1D (B, C, D, H, W) -> (B, C, L)
        x_HBflat = self.HB3D_Flat(x)
        # 2. 调整维度以适应 Mamba 的输入要求: (B, C, L) -> (B, L, C)
        x_HBflat = x_HBflat.transpose(-1, -2)
        # 3. Normalization 和 Mamba 前向计算
        x_norm = self.norm(x_HBflat)
        x_mamba = self.mamba(x_norm)
        # 4. 调整回 Channel 优先并恢复 3D 结构: (B, L, C) -> (B, C, L) -> (B, C, S, S, S)
        out = self.HB3D_Reshape(x_mamba.transpose(-1, -2), S)

        # 加上残差连接
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
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3], hilbertMaps=None):
        super().__init__()

        self.hilbertMaps = hilbertMaps
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
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i], hilbertMaps=hilbertMaps)
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


class hilbert3dMamba(nn.Module):
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

        # 初始化 hilbert的hilbertMap
        hilbertMaps = Hilbert3DMapper().hilbert3dMaps

        self.spatial_dims = spatial_dims
        self.vim = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                hilbertMaps=hilbertMaps
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
        outs = self.vim(x_in)
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

        return self.out(out)




