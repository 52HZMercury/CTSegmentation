import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

from monai.data import (
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch
)
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    EnsureTyped,
    SaveImage  # <--- [修改点1] 引入非字典版本的 SaveImage
)

# 假设这些模块都在你的项目中
from models.getmodel import create_model
from data.Augmentation import val_transforms


def run_inference(config_path):
    # 1. 加载配置
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 设置设备
    device = torch.device(config['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 准备数据
    datasets = config['data']['split_json']
    val_files = load_decathlon_datalist(
        datasets,
        is_segmentation=True,
        # data_list_key=config['data']['validation_key']
        data_list_key=config['data']['datasets_key']
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 4. 初始化模型
    print("Building model...")
    model = create_model()
    model.to(device)
    model.eval()

    # 5. 定义后处理变换 (移除了 SaveImaged)
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ])

    # 注意：这里不再在循环外初始化 saver，因为 output_dir 需要动态变化

    # 6. 推理循环
    base_output_dir = "prediction"
    print(f"Start inference on {len(val_ds)} images...")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            val_inputs = batch["image"].to(device)

            # 推理
            roi_size = config['transforms']['resize']['spatial_size']
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )

            # 解包
            val_data_list = decollate_batch(batch)
            val_outputs_list = decollate_batch(val_outputs)

            for d, pred in zip(val_data_list, val_outputs_list):
                d["pred"] = pred

                # 执行后处理
                d = post_transforms(d)

                # --- 动态路径解析 ---
                original_path = d["image_meta_dict"]["filename_or_obj"]

                # 提取父文件夹名称 (如 "T059")
                subject_folder = os.path.basename(os.path.dirname(original_path))

                # 组合当前样本的特定输出路径
                current_output_dir = os.path.join(base_output_dir, subject_folder)

                if not os.path.exists(current_output_dir):
                    os.makedirs(current_output_dir)

                # --- [修改核心] 在循环内初始化 Saver ---
                saver = SaveImage(
                    output_dir=current_output_dir,  # 这里传入动态路径
                    output_postfix="seg",
                    output_ext=".nii.gz",
                    resample=False,
                    separate_folder=False,
                    print_log=True
                )

                # 调用保存 (此时不需要再传 output_dir)
                saver(d["pred"])

    print(f"Inference finished. Results saved to structure: {base_output_dir}/<SubjectID>/..._seg.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_inference(args.config)