import numpy as np
import yaml
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    Rand3DElasticd,
    AdjustContrastd,
    Lambdad
)

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

LABEL_KEY = config['data'].get('label_key', 'label')


# # 分段分割的
# # compose将transforms组合在一起
# # 图像尺寸为[1, 259, 223, 74]
# train_transforms = Compose(
#     [
#         # 加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
#         LoadImaged(keys=["image", "label", "all_lab"], image_only=False),
#         # 自动添加一个通道的维度，保证通道在第一维度
#         EnsureChannelFirstd(keys=["image", "label", "all_lab"]),
#         # 对图像进行一个方向变换，转为RAS坐标
#         Orientationd(keys=["image", "label", "all_lab"], axcodes="RAS"),
#         # 对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
#         Spacingd(
#             keys=["image", "label", "all_lab"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode']),
#         ),
#         # 对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         # 根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
#         CropForegroundd(keys=["image", "label", "all_lab"], source_key="label",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label", "all_lab"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#         # AdjustContrastd(keys=["image"], gamma = 2),
#         # 将图像裁剪为4个子图
#         RandCropByPosNegLabeld(
#             keys=["image", "label", "all_lab"],
#             label_key="label",
#             spatial_size=config['transforms']['rand_crop']['spatial_size'],
#             pos=config['transforms']['rand_crop']['pos'],
#             neg=config['transforms']['rand_crop']['neg'],  # 设置为0，因为我们没有背景样本
#             num_samples=config['transforms']['rand_crop']['num_samples'],
#             image_key="image",
#             image_threshold=0,
#         ),
#         # 随机旋转，按着0轴旋转，旋转概率为0.1
#         RandFlipd(
#             keys=["image", "label", "all_lab"],
#             spatial_axis=[0],
#             prob=0.5,
#         ),
#         # 随机旋转，按着1轴旋转，旋转概率为0.1
#         RandFlipd(
#             keys=["image", "label", "all_lab"],
#             spatial_axis=[1],
#             prob=0.5,
#         ),
#         # 随机旋转，按着2轴旋转，旋转概率为0.1
#         RandFlipd(
#             keys=["image", "label", "all_lab"],
#             spatial_axis=[2],
#             prob=0.5,
#         ),
#         # 随机旋转，概率为0.1，旋转次数为3
#         RandRotate90d(
#             keys=["image", "label", "all_lab"],
#             prob=0.5,
#             max_k=4,
#         ),
#         # 随机强度转换，强度偏移量为[-0.1, 0.1]，概率为0.5
#         RandShiftIntensityd(
#             keys=["image"],
#             offsets=config['transforms']['rand_shift_intensity']['offsets'],
#             prob=config['transforms']['rand_shift_intensity']['prob'],
#         ),
#     ]
# )
#
# val_transforms = Compose(
#     [
#         # 加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
#         LoadImaged(keys=["image", "label", "all_lab"], image_only=False),
#         # 自动添加一个通道的维度，保证通道在第一维度
#         EnsureChannelFirstd(keys=["image", "label", "all_lab"]),
#         # 对图像进行一个方向变换，转为RAS坐标
#         Orientationd(keys=["image", "label", "all_lab"], axcodes="RAS"),
#         # 对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
#         Spacingd(
#             keys=["image", "label", "all_lab"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode']),
#         ),
#         # 对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         # 根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
#         CropForegroundd(keys=["image", "label", "all_lab"], source_key="label",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label", "all_lab"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#     ]
# )

# # 多个标签的训练
# train_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label_1", "label_2", "label_3"], image_only=False),
#         EnsureChannelFirstd(keys=["image", "label_1", "label_2", "label_3"]),
#         Orientationd(keys=["image", "label_1", "label_2", "label_3"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label_1", "label_2", "label_3"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode']),
#         ),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "label_1", "label_2", "label_3"], source_key="label_3",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label_1", "label_2", "label_3"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#         RandCropByPosNegLabeld(
#             keys=["image", "label_1", "label_2", "label_3"],
#             label_key="label_3",
#             spatial_size=config['transforms']['rand_crop']['spatial_size'],
#             pos=config['transforms']['rand_crop']['pos'],
#             neg=config['transforms']['rand_crop']['neg'],
#             num_samples=config['transforms']['rand_crop']['num_samples'],
#             image_key="image",
#             image_threshold=0,
#         ),
#         RandFlipd(
#             keys=["image", "label_1", "label_2", "label_3"],
#             spatial_axis=[0],
#             prob=0.5,
#         ),
#         RandFlipd(
#             keys=["image", "label_1", "label_2", "label_3"],
#             spatial_axis=[1],
#             prob=0.5,
#         ),
#         RandFlipd(
#             keys=["image", "label_1", "label_2", "label_3"],
#             spatial_axis=[2],
#             prob=0.5,
#         ),
#         RandRotate90d(
#             keys=["image", "label_1", "label_2", "label_3"],
#             prob=0.5,
#             max_k=4,
#         ),
#         RandShiftIntensityd(
#             keys=["image"],
#             offsets=config['transforms']['rand_shift_intensity']['offsets'],
#             prob=config['transforms']['rand_shift_intensity']['prob'],
#         ),
#     ]
# )

# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label_1", "label_2", "label_3"], image_only=False),
#         EnsureChannelFirstd(keys=["image", "label_1", "label_2", "label_3"]),
#         Orientationd(keys=["image", "label_1", "label_2", "label_3"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label_1", "label_2", "label_3"],
#             pixdim=config['transforms']['spacing']['pixdim'],
#             mode=(config['transforms']['spacing']['mode']),
#         ),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=config['transforms']['scale_intensity']['a_min'],
#             a_max=config['transforms']['scale_intensity']['a_max'],
#             b_min=config['transforms']['scale_intensity']['b_min'],
#             b_max=config['transforms']['scale_intensity']['b_max'],
#             clip=config['transforms']['scale_intensity']['clip'],
#         ),
#         CropForegroundd(keys=["image", "label_1", "label_2", "label_3"], source_key="label_3",
#                         margin=config['transforms']['crop_foreground']['margin']),
#         ResizeWithPadOrCropd(keys=["image", "label_1", "label_2", "label_3"], spatial_size=config['transforms']['resize']['spatial_size'],
#                              mode=config['transforms']['resize']['mode']),
#     ]
# )


# 单标签的
def build_train_transforms(label_key=None):
    """Build training transforms for a label key (defaults to config data.label_key)."""
    label_key = label_key or LABEL_KEY
    return Compose(
        [
            LoadImaged(keys=["image", label_key], image_only=False),
            EnsureChannelFirstd(keys=["image", label_key]),
            # 强制二值化
            Lambdad(keys=[label_key], func=lambda x: (x > 0).to(dtype=x.dtype)),
            Orientationd(keys=["image", label_key], axcodes="RAS"),
            Spacingd(
                keys=["image", label_key],
                pixdim=config['transforms']['spacing']['pixdim'],
                mode=(config['transforms']['spacing']['mode'], "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config['transforms']['scale_intensity']['a_min'],
                a_max=config['transforms']['scale_intensity']['a_max'],
                b_min=config['transforms']['scale_intensity']['b_min'],
                b_max=config['transforms']['scale_intensity']['b_max'],
                clip=config['transforms']['scale_intensity']['clip'],
            ),


            CropForegroundd(keys=["image", label_key], source_key=label_key,
                            margin=config['transforms']['crop_foreground']['margin']),
            # 只 pad 不 crop：保证体积不小于随机裁剪的 ROI，尺寸超出时保持原样
            SpatialPadd(keys=["image", label_key],
                        spatial_size=config['transforms']['rand_crop']['spatial_size'],
                        mode="constant"),
            # 不要 ResizeWithPadOrCrop 到固定尺寸：尺寸超出时它是居中裁剪，
            # 会让血管两端永远进不了训练 patch。RandCropByPosNegLabeld 直接从完整前景框里采样即可。
            RandCropByPosNegLabeld(
                keys=["image", label_key],
                label_key=label_key,
                spatial_size=config['transforms']['rand_crop']['spatial_size'],
                pos=config['transforms']['rand_crop']['pos'],
                neg=config['transforms']['rand_crop']['neg'],
                num_samples=config['transforms']['rand_crop']['num_samples'],
                image_key="image",
                image_threshold=0,
            ),

            # 三类注释
            RandFlipd(
                keys=["image", label_key],
                spatial_axis=[0],
                prob=0.5,
            ),

            RandFlipd(
                keys=["image", label_key],
                spatial_axis=[1],
                prob=0.5,
            ),
            RandFlipd(
                keys=["image", label_key],
                spatial_axis=[2],
                prob=0.5,
            ),

            # 三类注释
            RandRotate90d(
                keys=["image", label_key],
                prob=0.5,
                max_k=4,
            ),

            RandShiftIntensityd(
                keys=["image"],
                offsets=config['transforms']['rand_shift_intensity']['offsets'],
                prob=config['transforms']['rand_shift_intensity']['prob'],
            ),
        ]
    )


def build_val_transforms(label_key=None):
    """Build validation transforms for a label key (defaults to config data.label_key)."""
    label_key = label_key or LABEL_KEY
    return Compose(
        [
            LoadImaged(keys=["image", label_key], image_only=False),
            EnsureChannelFirstd(keys=["image", label_key]),
            # 强制二值化
            Lambdad(keys=[label_key], func=lambda x: (x > 0).to(dtype=x.dtype)),
            Orientationd(keys=["image", label_key], axcodes="RAS"),
            Spacingd(
                keys=["image", label_key],
                pixdim=config['transforms']['spacing']['pixdim'],
                mode=(config['transforms']['spacing']['mode'], "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config['transforms']['scale_intensity']['a_min'],
                a_max=config['transforms']['scale_intensity']['a_max'],
                b_min=config['transforms']['scale_intensity']['b_min'],
                b_max=config['transforms']['scale_intensity']['b_max'],
                clip=config['transforms']['scale_intensity']['clip'],
            ),
            CropForegroundd(keys=["image", label_key], source_key=label_key,
                            margin=config['transforms']['crop_foreground']['margin']),
            # 验证/推理不做 ResizeWithPadOrCrop：尺寸超出时它会居中裁剪，
            # 把血管两端切掉，模型无法预测被裁掉的区域，Dice 也只反映裁剪窗口。
            # sliding_window_inference 本身可以处理任意尺寸，交给 Invertd 还原即可。
        ]
    )


# 兼容原有导入方式（trainer.py 等仍直接使用这两个实例，键取自配置文件）
train_transforms = build_train_transforms()
val_transforms = build_val_transforms()

