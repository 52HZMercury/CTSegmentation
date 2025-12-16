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
    AsDiscreted,  # 改用这个
    Compose,
    Invertd,
    SaveImaged,  # 改用这个
    EnsureTyped  # 改用这个
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
    # 注意：这里默认使用 validation_key 指定的数据集。
    # 如果你想推理 training 集，可以修改 data_list_key。
    datasets = config['data']['split_json']
    val_files = load_decathlon_datalist(
        datasets,
        is_segmentation=True,
        data_list_key=config['data']['validation_key']
        # data_list_key=config['data']['datasets_key']
    )

    # 使用 Dataset 而不是 CacheDataset，推理通常不需要缓存到内存
    val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # 推理通常逐个进行，batch_size=1 更安全
        shuffle=False,
        num_workers=0,  # 根据CPU情况调整
        pin_memory=False
    )

    # 4. 初始化模型并加载权重
    print("Building model...")
    model = create_model()

    model.to(device)
    model.eval()

    # 5. 定义后处理变换 (Post-processing)
    post_transforms = Compose([
        # 确保数据是 Tensor 类型 (针对字典)
        EnsureTyped(keys="pred"),

        # 反转变换 (输出仍为字典)
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

        # [修改点] 使用字典版本的 Argmax
        AsDiscreted(keys="pred", argmax=True),

        # [修改点] 使用字典版本的 SaveImage，指定保存 "pred" 这个 key
        SaveImaged(
            keys="pred",
            output_dir=os.path.join("prediction"),
            output_postfix="seg",
            output_ext=".nii.gz",
            resample=False,  # 已经通过 Invertd 还原了，这里不需要重采样
            separate_folder=False,  # 不为每个文件创建子文件夹
            print_log=True
        )
    ])

    # 6. 推理循环
    print(f"Start inference on {len(val_ds)} images...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            val_inputs = batch["image"].to(device)

            # 1. 执行推理
            roi_size = config['transforms']['resize']['spatial_size']
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )

            # 2. 解包原始数据 (List of Dicts)
            # decollate_batch 会把 GPU 上的 batch 拆分成单个样本，并处理好 meta_dict
            val_data_list = decollate_batch(batch)

            # 3. 解包预测结果 (List of Tensors)
            val_outputs_list = decollate_batch(val_outputs)

            # 4. 组合并执行后处理
            # 必须在一个循环里完成：赋值 -> 逆变换 -> 保存
            for d, pred in zip(val_data_list, val_outputs_list):
                # 关键步骤：先将预测结果放入字典，键名为 "pred"
                d["pred"] = pred

                # 现在字典里有 "pred" 了，可以安全调用 post_transforms
                # Invertd 会根据 d["pred"] 和 d["image_meta_dict"] 进行逆变换
                post_transforms(d)

    print("Inference finished. Results saved to:", os.path.join("prediction"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")

    args = parser.parse_args()

    run_inference(args.config)
