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
    RandRotate90d,
    Rand3DElasticd,
    ResizeWithPadOrCropd,
    AdjustContrastd
)

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# compose将transforms组合在一起
# 图像尺寸为[1, 259, 223, 74]
train_transforms = Compose(
    [
        # 加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"], image_only=False),
        # 自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        # 对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=(config['transforms']['spacing']['mode']),
        ),
        # 对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        # 根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="label", margin=config['transforms']['crop_foreground']['margin']),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config['transforms']['resize']['spatial_size'], mode=config['transforms']['resize']['mode']),
        # AdjustContrastd(keys=["image"], gamma = 2),
        # 将图像裁剪为4个子图
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config['transforms']['rand_crop']['spatial_size'],
            pos=config['transforms']['rand_crop']['pos'],
            neg=config['transforms']['rand_crop']['neg'],  # 设置为0，因为我们没有背景样本
            num_samples=config['transforms']['rand_crop']['num_samples'],
            image_key="image",
            image_threshold=0,
        ),
        # 随机旋转，按着0轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.5,
        ),
        # 随机旋转，按着1轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.5,
        ),
        # 随机旋转，按着2轴旋转，旋转概率为0.1
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.5,
        ),
        # 随机旋转，概率为0.1，旋转次数为3
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=4,
        ),
        # 随机强度转换，强度偏移量为[-0.1, 0.1]，概率为0.5
        RandShiftIntensityd(
            keys=["image"],
            offsets=config['transforms']['rand_shift_intensity']['offsets'],
            prob=config['transforms']['rand_shift_intensity']['prob'],
        ),
    ]
)

val_transforms = Compose(
    [
        # 加载图片的值和元数据，参数keys是data_dicts中设置的keys，表示对image还是label做变换
        LoadImaged(keys=["image", "label"], image_only=False),
        # 自动添加一个通道的维度，保证通道在第一维度
        EnsureChannelFirstd(keys=["image", "label"]),
        # 对图像进行一个方向变换，转为RAS坐标
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 对图像进行重采样，体素间距重采样为[1.5, 1.5, 2.0]
        Spacingd(
            keys=["image", "label"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=(config['transforms']['spacing']['mode']),
        ),
        # 对图像值强度进行归一化，由a的范围归一到b，，不在a范围的值设置为0
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        # 根据key所指定的大于0的部分，对于图像的有效部分进行裁剪
        CropForegroundd(keys=["image", "label"], source_key="label", margin=config['transforms']['crop_foreground']['margin']),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config['transforms']['resize']['spatial_size'], mode=config['transforms']['resize']['mode']),
    ]
)