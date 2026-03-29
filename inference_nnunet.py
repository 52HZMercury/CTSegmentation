import os
import torch
import yaml
import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    ResizeWithPadOrCropd, Invertd
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import BasicUnet
# 假设你的模型创建函数在这里
from models.getmodel import create_model


def main():
    # 1. 配置路径
    config_path = "config/config.yaml"
    weight_path = "/workdir1.8t/cn24/program/CT_Seg/logs/exp_95/checkpoint/best_metric_model_0.7182.pth"
    input_folder = "/workdir1.8t/cn24/data/PVT_huaxi_pro/image_pro"
    output_folder = "/workdir1.8t/cn24/data/PVT_huaxi_pro/CA"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. 定义推理变换 (参考你的 val_transforms)
    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        # 注意：推理时不建议使用 CropForegroundd，除非你有对应的 Inverse 变换
        # 这里为了保持一致，我们使用同样的尺寸处理
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=config['transforms']['resize']['spatial_size'],
            mode="constant"
        ),
    ])

    # 4. 加载数据
    img_files = sorted(glob.glob(os.path.join(input_folder, "*.nii.gz")))
    data_dicts = [{"image": f} for f in img_files]
    ds = Dataset(data=data_dicts, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    # 5. 初始化模型并加载权重
    model = create_model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    print(f"开始推理，共 {len(img_files)} 个文件...")

    # 6. 推理循环
    with torch.no_grad():
        for batch_data in tqdm(loader):
            inputs = batch_data["image"].to(device)
            # 使用滑动窗口推理，ROI size 参考你的训练配置
            roi_size = config['transforms']['rand_crop']['spatial_size']
            outputs = sliding_window_inference(inputs, roi_size, 4, model)

            # 取最大概率类别 (Argmax)
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

            # 7. 保存结果
            # 为了保持原始的 Header 信息（Spacing, Origin），我们读取原图作为模板
            raw_img_path = batch_data["image_meta_dict"]["filename_or_obj"][0]
            file_name = os.path.basename(raw_img_path)

            # 使用 nibabel 恢复原始尺寸并保存
            original_nib = nib.load(raw_img_path)
            # 注意：这里的 outputs 尺寸可能因为变换与原图不同，
            # 如果需要完美对齐，需要使用 monai.transforms.Invertd

            save_path = os.path.join(output_folder, file_name)
            new_seg = nib.Nifti1Image(outputs.astype(np.uint8), original_nib.affine, original_nib.header)
            nib.save(new_seg, save_path)

    print(f"推理完成！结果保存在: {output_folder}")


if __name__ == "__main__":
    main()