import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from nibabel.processing import resample_from_to
from monai.data import DataLoader, Dataset, decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureTyped,
    Invertd,
)
from tqdm import tqdm

from data.Augmentation import build_val_transforms
from models.getmodel import load_nnunet_model


DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_SPLIT_JSON = "metadata/Takayasu_common9.json"
DEFAULT_LABEL_KEY = "label_DIS_Abdominal_Infrarenal"
DEFAULT_PLANS_PATH = "weights/plans.json"
DEFAULT_DATASET_JSON_PATH = "weights/dataset.json"
DEFAULT_WEIGHT_PATH = (
    "/workdir2/cn24/program/CT_Seg/logs/exp_260721-1032/checkpoint/"
    "best_metric_model_0.8284.pth"
)
DEFAULT_OUTPUT_DIR = "/workdir2/cn24/program/CT_Seg/inference/ours/DIS_Abdominal_Infrarenal"


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _image_stem(image_path):
    name = os.path.basename(image_path)
    return name.replace(".nii.gz", "").replace(".nii", "")


def _case_id(image_path, duplicated_stems=()):
    """Use the file stem, falling back to the parent directory when stems collide.

    Datasets such as arterial_phase.nii.gz store one image per case directory, so
    the file name alone is not unique. neck_STA stores all images flat with unique
    names, where the parent directory ("neck") would collapse every case.
    """
    stem = _image_stem(image_path)
    if stem in duplicated_stems:
        parent = os.path.basename(os.path.dirname(image_path))
        return parent or stem
    return stem


def _dice_value(dice_metric, pred_labels, label, num_classes, device):
    pred_onehot = AsDiscrete(to_onehot=num_classes)(pred_labels).unsqueeze(0).to(device)
    label_onehot = AsDiscrete(to_onehot=num_classes)(label).unsqueeze(0).to(device)
    dice_metric(y_pred=pred_onehot, y=label_onehot)
    value = dice_metric.aggregate().item()
    dice_metric.reset()
    return value


def _pred_affine(pred, fallback):
    """Invertd 输出若是 MetaTensor，其 affine 才和数组所在的网格匹配。"""
    affine = getattr(pred, "affine", None)
    if affine is None:
        meta = getattr(pred, "meta", None) or {}
        affine = meta.get("affine")
    if affine is None:
        return fallback
    if hasattr(affine, "detach"):
        affine = affine.detach().cpu().numpy()
    return np.asarray(affine, dtype=np.float64)


def _save_prediction(pred, reference_image_path, save_path, target_image_path=None):
    pred_array = np.squeeze(pred.detach().cpu().numpy()).astype(np.uint8)
    reference = nib.load(reference_image_path)
    pred_img = nib.Nifti1Image(pred_array, _pred_affine(pred, reference.affine))

    # 以 label 的网格为准：网格不一致时用最近邻重采样过去，保证和 label 在 3D 空间重合
    target = nib.load(target_image_path) if target_image_path else reference
    if pred_img.shape != target.shape or not np.allclose(pred_img.affine, target.affine, atol=1e-3):
        print(
            f"[save] {os.path.basename(save_path)}: pred grid {pred_img.shape} / "
            f"affine differs from label grid {target.shape}, resampling to label grid"
        )
        pred_img = resample_from_to(pred_img, target, order=0)

    seg = nib.Nifti1Image(pred_img.get_fdata().astype(np.uint8), target.affine, target.header)
    nib.save(seg, save_path)


def run_inference(
    config_path,
    split_json,
    plans_path,
    dataset_json_path,
    weight_path,
    output_dir,
    label_key=DEFAULT_LABEL_KEY,
):
    config = _load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config["device"]["cuda_device"] if torch.cuda.is_available() else "cpu")
    validation_key = config["data"]["validation_key"]
    label_key = label_key or DEFAULT_LABEL_KEY
    # 必须用同一个 label_key 构建 transforms，否则 Dataset/Invertd 会按配置里的键加载
    val_transforms = build_val_transforms(label_key)
    num_classes = int(config["training"].get("num_class", config["model"]["out_channels"]))
    excel_path = os.path.join(output_dir, "validation_dice.xlsx")

    val_files = load_decathlon_datalist(
        split_json,
        is_segmentation=True,
        data_list_key=validation_key,
    )
    for item in val_files:
        if label_key not in item:
            available = sorted(k for k in item if k.startswith("label"))
            raise KeyError(
                f"{label_key!r} not found in validation item "
                f"(available label keys: {available}); "
                "set data.label_key in config/config.yaml"
            )

    stems = [_image_stem(item["image"]) for item in val_files]
    duplicated_stems = {stem for stem in stems if stems.count(stem) > 1}

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["validation"].get("batch_size", 1),
        shuffle=False,
        num_workers=config["validation"].get("num_workers", 0),
        pin_memory=config["validation"].get("pin_memory", False),
    )

    print(f"Using device: {device}")
    print(f"Using split json: {split_json}")
    print(f"Using label key: {label_key}")
    print(f"Using nnU-Net plans: {plans_path}")
    print(f"Using nnU-Net dataset json: {dataset_json_path}")
    print(f"Loading nnU-Net weights from: {weight_path}")
    model = load_nnunet_model(
        weight_path,
        plans_path=plans_path,
        dataset_json_path=dataset_json_path,
    ).to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    roi_size = tuple(config["transforms"]["rand_crop"]["spatial_size"])

    post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                # 对 logits 做线性插值再 argmax，边界能落在亚体素位置；
                # 用最近邻会把粗网格（如 1.62x1.62x3.2）的预测原样放大成砖块，3D 表面全是台阶。
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
        ]
    )

    results = []
    print(f"Start inference on {len(val_ds)} validation cases...")
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc="Inference"):
            inputs = batch["image"].to(device)
            labels = batch[label_key].to(device)
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                # nnU-Net 用高斯权重融合滑窗；MONAI 默认 "constant" 是等权平均，窗口接缝处会出现台阶
                mode="gaussian",
            )

            pred_labels = torch.argmax(logits, dim=1, keepdim=True)
            batch_dicts = decollate_batch(batch)
            pred_dicts = decollate_batch(logits)

            for idx, (data_dict, pred_logits) in enumerate(zip(batch_dicts, pred_dicts)):
                image_path = data_dict["image_meta_dict"]["filename_or_obj"]
                case_id = _case_id(image_path, duplicated_stems)
                dice = _dice_value(
                    dice_metric=dice_metric,
                    pred_labels=pred_labels[idx],
                    label=labels[idx],
                    num_classes=num_classes,
                    device=device,
                )

                label_path = data_dict[f"{label_key}_meta_dict"]["filename_or_obj"]

                data_dict["pred"] = pred_logits
                data_dict = post_transforms(data_dict)
                pred_save_path = os.path.join(output_dir, f"{case_id}_pred.nii.gz")
                _save_prediction(data_dict["pred"], image_path, pred_save_path, label_path)

                results.append(
                    {
                        "case_id": case_id,
                        "image": image_path,
                        "label_key": label_key,
                        "label": label_path,
                        "prediction": pred_save_path,
                        "dice": round(float(dice), 6),
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        df.loc[len(df)] = {
            "case_id": "MEAN",
            "image": "",
            "label_key": label_key,
            "label": "",
            "prediction": "",
            "dice": round(float(df["dice"].mean()), 6),
        }
    df.to_excel(excel_path, index=False)

    print(f"Inference results saved to: {output_dir}")
    print(f"Dice Excel saved to: {excel_path}")
    if not df.empty:
        print(f"Mean Dice: {df.iloc[-1]['dice']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="nnU-Net inference for validation data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON, help="Path to  split JSON")
    parser.add_argument("--plans", default=DEFAULT_PLANS_PATH, help="Path to nnU-Net plans.json")
    parser.add_argument(
        "--dataset-json",
        default=DEFAULT_DATASET_JSON_PATH,
        help="Path to nnU-Net dataset.json",
    )
    parser.add_argument("--weight", default=DEFAULT_WEIGHT_PATH, help="Path to checkpoint weights")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Directory to save predictions and Excel")
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help="Validation label key, e.g. label / label_1 / label_2 / label_3. Defaults to DEFAULT_LABEL_KEY.",
    )
    args = parser.parse_args()

    run_inference(
        args.config,
        args.split_json,
        args.plans,
        args.dataset_json,
        args.weight,
        args.output,
        args.label_key,
    )


if __name__ == "__main__":
    main()
