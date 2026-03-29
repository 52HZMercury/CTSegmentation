import os
import torch
import yaml
import glob
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, ResizeWithPadOrCropd,
    Invertd, AsDiscreted, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from models.getmodel import create_model


def main():
    # --- 1. 路径设置 ---
    config_path = "config/config.yaml"
    weight_path = "/workdir2/cn24/program/CT_Seg/logs/exp_116/checkpoint/best_metric_model_0.5974.pth"

    input_folder = "/workdir2/cn24/data/SCU_CA_CAC/image"
    label_folder = "/workdir2/cn24/data/SCU_CA_CAC/CA"  # CA 标签
    cac_folder = "/workdir2/cn24/data/SCU_CA_CAC/CAC"  # 新增 CAC 标签

    output_folder = "/workdir2/cn24/data/SCU_CA_CAC/CAC_pred"
    excel_save_path = os.path.join("/workdir2/cn24/data/SCU_CA_CAC", "CAC_inference_dice_results.xlsx")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # --- 2. 准备数据列表 ---
    images = sorted(glob.glob(os.path.join(input_folder, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_folder, "*.nii.gz")))
    cacs = sorted(glob.glob(os.path.join(cac_folder, "*.nii.gz")))

    # 确保文件一一对应 (假设文件名一致)
    data_dicts = [
        {"image": img, "label": lbl, "cac": cac}
        for img, lbl, cac in zip(images, labels, cacs)
    ]

    # --- 3. 定义变换 ---
    # 在 keys 中加入 "cac"
    infer_transforms = Compose([
        LoadImaged(keys=["image", "label", "cac"]),
        EnsureChannelFirstd(keys=["image", "label", "cac"]),
        Orientationd(keys=["image", "label", "cac"], axcodes="RAS"),
        # 标签通常使用 nearest 插值以保持类别属性
        Spacingd(
            keys=["image", "label", "cac"],
            pixdim=config['transforms']['spacing']['pixdim'],
            mode=("bilinear", "nearest", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=0.0, b_max=1.0, clip=True
        ),
    ])

    ds = Dataset(data=data_dicts, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    # --- 4. 加载模型 ---
    model = create_model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 逆向变换：将所有东西转回原始空间
    post_transforms = Compose([
        Invertd(
            keys=["pred", "label", "cac"],  # 同时也对 label 和 cac 进行逆变换
            transform=infer_transforms,
            orig_keys="image",
            meta_keys=["pred_meta_dict", "label_meta_dict", "cac_meta_dict"],
            orig_meta_keys="image_meta_dict",
            to_tensor=True,
        ),
        # 此时先只做 argmax 得到类别索引(0, 1)，不做 One-hot，方便后面相乘
        AsDiscreted(keys="pred", argmax=True),
    ])

    results = []

    # --- 5. 推理与评估 ---
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Inference"):
            inputs = batch_data["image"].to(device)
            roi_size = config['transforms']['rand_crop']['spatial_size']

            batch_data["pred"] = sliding_window_inference(
                inputs, roi_size, 4, model
            )

            # 执行逆向变换（现在 pred, label, cac 都在原始尺寸了）
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)][0]

            # 1. 获取原始空间的预测掩码 (已经是 [1, H, W, D])
            pred_mask = batch_data["pred"].to(device)

            # 2. 获取原始空间的 CA 和 CAC 掩码，并相乘得到交集
            # 将它们都转到 GPU 上再进行相乘
            label_tensor = batch_data["label"].to(device)
            cac_tensor = batch_data["cac"].to(device)
            final_label_mask = label_tensor * cac_tensor

            # 3. 转换成 One-hot 格式以适配 DiceMetric (通常需要 [C, H, W, D])
            # 假设是 2 分类（背景 + 目标）
            y_pred_onehot = AsDiscrete(to_onehot=2)(pred_mask).unsqueeze(0).to(device)
            y_true_onehot = AsDiscrete(to_onehot=2)(final_label_mask).unsqueeze(0).to(device)

            # 4. 计算 Dice
            dice_metric(y_pred=y_pred_onehot, y=y_true_onehot)
            curr_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            # 获取文件名并保存
            file_path = batch_data["image_meta_dict"]["filename_or_obj"]
            file_id = os.path.basename(file_path).replace(".nii.gz", "")

            results.append({"File_ID": file_id, "Dice": curr_dice})

            # 保存预测结果 (可选)
            # save_path = os.path.join(output_folder, f"{file_id}_pred.nii.gz")
            # ... (此处省略保存 nib 文件的代码)

    # --- 6. 保存结果到 Excel ---
    df = pd.DataFrame(results)
    df.to_excel(excel_save_path, index=False)
    print(f"Results saved to {excel_save_path}")
    print(f"Mean Dice: {df['Dice'].mean():.4f}")


if __name__ == "__main__":
    main()