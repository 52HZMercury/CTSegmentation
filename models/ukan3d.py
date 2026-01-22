import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
# from monai.utils import alias, deprecated_arg, export

# from model.utils.kan import KANBlock
from .utils.kan import KANBlock

class UNet3DbottleKAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(1, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128])
        # self.encoder5 = self._create_encoder_block(128, [256, 256, 256])

        self.bottleKAN = KANBlock(dim=128, num_heads=8)
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 2, True)
        
        self.upconv5 = self._create_up_conv(128, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        _, _, h, w, d = x.shape
        
        if (h%16 != 0) or (w%16 != 0) or (d%16 != 0):
            raise ValueError(f"Invalid volume size ({h}, {w}, {d}). The dimension need to be divisible by 16.")
        
        x1 = self.encoder1(x)
        x  = self.maxpool(x1)
        
        x2 = self.encoder2(x)
        x  = self.maxpool(x2)
        
        x3 = self.encoder3(x)
        x  = self.maxpool(x3)
        
        x4 = self.encoder4(x)
        x  = self.maxpool(x4)
        
        # x = self.encoder5(x)  # [1, 128, 14, 14, 4] -> [1, 256, 7 ,7, 2]
        x = self.bottleKAN(x)
        
        x = self.upconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        return x
    
    def _create_encoder_block(
        self,
        in_channel,
        channels,
        down_sampling=False,
        ):
        
        def _create_residual_unit(
            in_channels, out_channels, strides,
            ):
            return ResidualUnit(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )
        
        _channels = [in_channel, *channels]
        
        units = []
        for i in range(len(channels) - 1):
            units.append(_create_residual_unit(_channels[i], _channels[i+1], 1))
        units.append(
            _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
        )
        
        return nn.Sequential(*units)
    
    def _create_decoder_block(
        self,
        in_channels,
        out_channels,
        is_top,
    ):
        res_unit = ResidualUnit(
            3,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=is_top,
            adn_ordering="NDA",
        )
        
        return res_unit
    
    def _create_up_conv(
        self,
        in_channels,
        out_channels,
    ):
        return Convolution(
            3,
            in_channels,
            out_channels,
            strides=2,
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            is_transposed=True,
            adn_ordering="NDA",
        )

class UNet3DSmall(UNet3DbottleKAN):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128])
        # self.encoder5 = self._create_encoder_block(128, [256, 256])
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 1, True)
        
        self.upconv5 = self._create_up_conv(128, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)



# if __name__ == "__main__":
#     # 初始化模型
#     model = UNet3DbottleKAN().cuda(1)  # 或者使用 UNet3DSmall()
#     # model = OnlyUKAN3D().cuda(1)  # 或者使用 UNet3DSmall()
#     # 伪造输入数据：假设输入形状为 (batch_size=1, channels=3, depth=128, height=128, width=128)
#     input = torch.randn(1, 3, 112, 112, 16).cuda(1)
#
#     output = model(input)
#
#     from thop import profile
#
#     flops, params = profile(model, inputs=(input,))
#     print('Flops: ', flops, ', Params: ', params)
#     print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
#
#     print(f"输出形状: {output.shape}")


    # step1 加载模型权重
    # checkpoint_path = r'E:\MyExperiment\A-Echo\SimLVSeg\lightning_logs\auther_version_135_SI\checkpoints\epoch=20-step=19339.ckpt'
    # checkpoint = torch.load(checkpoint_path, map_location='cuda:1', weights_only=True)
    #
    # # 获取 state_dict
    # state_dict = checkpoint['state_dict']
    #
    # # 移除 'model.' 前缀
    # new_state_dict = {}
    # for key in state_dict:
    #     new_key = key.replace("model.", "")  # 移除 'model.' 前缀
    #     new_state_dict[new_key] = state_dict[key]

    # # step2 初始化模型并加载权重
    # model = UNet3D()
    # model.load_state_dict(new_state_dict)
    #
    # # 将模型加载到 GPU 1
    # model = model.cuda(1)
    #


    # step3 加载真实数据
    # from simlvseg.model.utils.img2tensor import video_to_tensor
    #
    # video_tensor = video_to_tensor(
    #     r'E:\MyExperiment\A-Echo\SimLVSeg\data\EchoNet-Dynamic\Videos\0X1A0A263B22CCD966.avi')
    # input = video_tensor[:, :, :, :, 0:32].cuda(1)

    # step4 前向传播
    # output = model(input)
    #
    # from thop import profile
    #
    # flops, params = profile(model, inputs=(input,))
    # print('Flops: ', flops, ', Params: ', params)
    # print('FLOPs&Params: ' + 'GFLOPs: %.2f G, Params: %.2f MB' % (flops / 1e9, params / 1e6))
    #
    # 打印输出形状
    # print(f"输出形状: {output.shape}")

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
    model = UNet3DbottleKAN().to(device)

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