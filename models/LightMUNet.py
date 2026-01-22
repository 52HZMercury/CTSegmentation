from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba

class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation3D, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv3d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool3d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class SqueezeAndExciteFusionAdd3D(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd3D, self).__init__()

        self.se_1 = SqueezeAndExcitation3D(channels_in, activation=activation)
        self.se_2 = SqueezeAndExcitation3D(channels_in, activation=activation)

    def forward(self, se1, se2):
        se1 = self.se_1(se1)
        se2 = self.se_2(se2)
        out = se1 + se2
        return out

def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class ResUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x

# 论文地址：https://arxiv.org/pdf/2303.16900
# 论文：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class InceptionDWConv3d(nn.Module):
    """ Inception depthwise convolution for 3D data
    """

    def __init__(self, in_channels, cube_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hwd = nn.Conv3d(gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc)
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2),
                                   groups=gc)
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0),
                                   groups=gc)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0),
                                   groups=gc)
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1,
        )

class LightMUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.channel_att_de = SqueezeAndExciteFusionAdd3D(channels_in=64)
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.channel_att_de(x, x)
        x = self.decode(x, down_x)

        return x


# if __name__ == "__main__":
#     from torchsummary import summary
#
#     # 查看模型参数量
#     model = LightMUNet(in_channels=3,
#                        out_channels=1)
#     # 将模型加载到 GPU 0
#     model = model.cuda(0)
#     # summary(model, input_size=(3, 128, 128, 128))
#     # ... existing code ...
#     from thop import profile
#
#     # 使用随机张量作为输入示例，避免将内置函数 input 传给模型
#     x = torch.randn(1, 3, 128, 128, 16).cuda(0)
#     model.eval()
#     flops, params = profile(model, inputs=(x,))
#     print('Flops: ', flops, ', Params: ', params)
#     print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))




def main():
    import os
    import time
    try:
        from thop import profile
        from thop import clever_format
    except ImportError:
        print("Error: 'thop' is not installed. Please run: pip install thop")
        return

    # 1. Configuration
    # NOTE: Update these parameters to match exactly what you used during training
    INPUT_SIZE = (1, 3, 128, 128, 16)  # (Batch, Channels, H, W, D)
    # CKPT_PATH = "/workdir1/cn24/program/SimLVSeg/lightning_logs/version_368/checkpoints/epoch=22-step=14260.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing model...")
    # Ensure these init parameters match your training config
    model = LightMUNet(in_channels=3,out_channels=1).to(device)

    model.eval()

    # 4. Create Dummy Input
    # Mamba usually requires inputs on CUDA
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # 5. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 6. Calculate FLOPs
    print("-" * 30)
    print("Calculating FLOPs... (this might take a moment)")
    try:
        # thop.profile returns (flops, params)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, total_params], "%.3f")
        print(f"GFLOPs: {flops / 1e9:.3f} G")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")

    # 7. Calculate FPS
    print("-" * 30)
    print("Calculating FPS...")
    num_iterations = 50
    warmup = 10

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_iterations / total_time

    print(f"Average time per inference: {total_time / num_iterations:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print("-" * 30)


if __name__ == '__main__':
    main()