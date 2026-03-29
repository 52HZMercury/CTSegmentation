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
    # ==========================================
    # 0. 开关配置：是否加载 Label 进行 Dice 计算
    # ==========================================
    USE_LABEL = False  # 设置为 True 则计算 Dice 并保存表格；设置为 False 则仅做推理

    # 1. 配置路径
    config_path = "config/config.yaml"
    weight_path = "/workdir2/cn24/program/CT_Seg/logs/exp_116/checkpoint/best_metric_model_0.5974.pth"
    input_folder = "/workdir2/cn24/data/30daysSuccess/image"
    output_folder = "/workdir2/cn24/data/30daysSuccess/CAC_pred"

    if USE_LABEL:
        label_folder = "/workdir2/cn24/data/30daysSuccess/CAC"
        excel_save_path = os.path.join("/workdir2/cn24/data/30daysSuccess", "CA_inference_dice_results.xlsx")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 3. 动态设置 Transform 的 keys
    # 根据 USE_LABEL 决定是否在 Transform 中处理 "label"
    data_keys = ["image", "label"] if USE_LABEL else ["image"]

    # 定义推理变换
    infer_transforms = Compose([
        LoadImaged(keys=data_keys),
        EnsureChannelFirstd(keys=data_keys),
        Orientationd(keys=data_keys, axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=config['transforms']['spacing']['pixdim'], mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config['transforms']['scale_intensity']['a_min'],
            a_max=config['transforms']['scale_intensity']['a_max'],
            b_min=config['transforms']['scale_intensity']['b_min'],
            b_max=config['transforms']['scale_intensity']['b_max'],
            clip=config['transforms']['scale_intensity']['clip'],
        ),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=config['transforms']['resize']['spatial_size'],
                             mode="constant"),
    ])

    # 4. 动态定义逆转变换
    post_transforms_list = [
        Invertd(keys="pred", transform=infer_transforms, orig_keys="image", meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict", to_tensor=True),
        AsDiscreted(keys="pred", argmax=True)
    ]
    # 只有在使用 label 时，才将 label 逆转回原始空间
    if USE_LABEL:
        post_transforms_list.append(
            Invertd(keys="label", transform=infer_transforms, orig_keys="label", meta_keys="label_meta_dict",
                    orig_meta_keys="label_meta_dict", to_tensor=True)
        )
    post_transforms = Compose(post_transforms_list)

    # 5. 数据加载与指标初始化
    img_files = sorted(glob.glob(os.path.join(input_folder, "*.nii.gz")))

    # 动态构建数据字典
    if USE_LABEL:
        data_dicts = [{"image": f, "label": os.path.join(label_folder, os.path.basename(f))} for f in img_files]
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        post_discrete = AsDiscrete(to_onehot=config['model']['out_channels'])
        results_list = []
    else:
        data_dicts = [{"image": f} for f in img_files]

    ds = Dataset(data=data_dicts, transform=infer_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    # 6. 模型初始化
    model = create_model().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval()

    # 7. 推理循环
    with torch.no_grad():
        for batch_data in tqdm(loader):
            inputs = batch_data["image"].to(device)
            roi_size = config['transforms']['rand_crop']['spatial_size']

            # 推理
            batch_data["pred"] = sliding_window_inference(inputs, roi_size, 4, model)

            # --- 新增：把所有需要的 tensor 移回 CPU，释放 GPU 压力 ---
            batch_data["pred"] = batch_data["pred"].cpu()
            if isinstance(batch_data["image"], torch.Tensor):
                batch_data["image"] = batch_data["image"].cpu()
            if "label" in batch_data and isinstance(batch_data["label"], torch.Tensor):
                batch_data["label"] = batch_data["label"].cpu()

            # 可选：清空一下 GPU 缓存碎片
            torch.cuda.empty_cache()

            # 后处理还原空间 (MONAI 内部根据 post_transforms_list 自动处理对应的 key)
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            # 提取 pred 和文件名
            pred_final = batch_data[0]["pred"].to(device)
            file_name = os.path.basename(batch_data[0]["image_meta_dict"]["filename_or_obj"])

            # 如果启用 label，则计算 Dice
            if USE_LABEL:
                label_final = batch_data[0]["label"].to(device)

                p_onehot = [post_discrete(pred_final)]
                l_onehot = [post_discrete(label_final)]
                dice_metric(y_pred=p_onehot, y=l_onehot)

                current_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                results_list.append({"ID": file_name, "Dice": round(current_dice, 4)})

            # 无论是否有 label，都保存 Nifti 预测结果
            original_nib = nib.load(batch_data[0]["image_meta_dict"]["filename_or_obj"])
            new_seg = nib.Nifti1Image(pred_final.squeeze().cpu().numpy().astype(np.uint8),
                                      original_nib.affine, original_nib.header)
            nib.save(new_seg, os.path.join(output_folder, file_name))

    # 8. 结尾输出判断
    if USE_LABEL:
        df = pd.DataFrame(results_list)
        df.loc[len(df)] = {"ID": "AVERAGE", "Dice": round(df["Dice"].mean(), 4)}
        df.to_excel(excel_save_path, index=False)
        print(f"完成！已保存预测结果与 Dice 表格，平均 Dice: {df.iloc[-1]['Dice']}")
    else:
        print("完成！未加载 Label，已成功保存所有预测的 NIfTI 文件。")


if __name__ == "__main__":
    main()