import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
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

from data.Augmentation import val_transforms
from models.getmodel import load_nnunet_model


DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_WEIGHT_PATH = (
    "/workdir2/cn24/program/CT_Seg/logs/exp_260428-1500/checkpoint/"
    "best_metric_model_0.8177.pth"
)
DEFAULT_OUTPUT_DIR = "/workdir2/cn24/program/CT_Seg/logs/exp_260428-1500/inference"


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _case_id(image_path):
    """Use parent folder as case id for arterial_phase.nii.gz style datasets."""
    parent = os.path.basename(os.path.dirname(image_path))
    name = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
    return parent or name


def _dice_value(dice_metric, pred_labels, label, num_classes, device):
    pred_onehot = AsDiscrete(to_onehot=num_classes)(pred_labels).unsqueeze(0).to(device)
    label_onehot = AsDiscrete(to_onehot=num_classes)(label).unsqueeze(0).to(device)
    dice_metric(y_pred=pred_onehot, y=label_onehot)
    value = dice_metric.aggregate().item()
    dice_metric.reset()
    return value


def _save_prediction(pred, reference_image_path, save_path):
    pred_array = pred.detach().cpu().numpy()
    pred_array = np.squeeze(pred_array).astype(np.uint8)
    reference = nib.load(reference_image_path)
    seg = nib.Nifti1Image(pred_array, reference.affine, reference.header)
    nib.save(seg, save_path)


def run_inference(config_path, weight_path, output_dir):
    config = _load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config["device"]["cuda_device"] if torch.cuda.is_available() else "cpu")
    split_json = config["data"]["split_json"]
    validation_key = config["data"]["validation_key"]
    num_classes = int(config["training"].get("num_class", config["model"]["out_channels"]))
    excel_path = os.path.join(output_dir, "validation_dice.xlsx")

    val_files = load_decathlon_datalist(
        split_json,
        is_segmentation=True,
        data_list_key=validation_key,
    )
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["validation"].get("batch_size", 1),
        shuffle=False,
        num_workers=config["validation"].get("num_workers", 0),
        pin_memory=config["validation"].get("pin_memory", False),
    )

    print(f"Using device: {device}")
    print(f"Loading nnU-Net weights from: {weight_path}")
    model = load_nnunet_model(weight_path).to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
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
                nearest_interp=True,
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
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            pred_labels = torch.argmax(logits, dim=1, keepdim=True)
            batch_dicts = decollate_batch(batch)
            pred_dicts = decollate_batch(logits)

            for idx, (data_dict, pred_logits) in enumerate(zip(batch_dicts, pred_dicts)):
                image_path = data_dict["image_meta_dict"]["filename_or_obj"]
                case_id = _case_id(image_path)
                dice = _dice_value(
                    dice_metric=dice_metric,
                    pred_labels=pred_labels[idx],
                    label=labels[idx],
                    num_classes=num_classes,
                    device=device,
                )

                data_dict["pred"] = pred_logits
                data_dict = post_transforms(data_dict)
                pred_save_path = os.path.join(
                    output_dir, f"{case_id}_pred.nii.gz"
                )
                _save_prediction(data_dict["pred"], image_path, pred_save_path)

                results.append(
                    {
                        "case_id": case_id,
                        "image": image_path,
                        "label": data_dict["label_meta_dict"]["filename_or_obj"],
                        "prediction": pred_save_path,
                        "dice": round(float(dice), 6),
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        df.loc[len(df)] = {
            "case_id": "MEAN",
            "image": "",
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
    parser = argparse.ArgumentParser(description="nnU-Net validation inference with Dice export.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT_PATH, help="Path to checkpoint weights")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Directory to save predictions and Excel")
    args = parser.parse_args()

    run_inference(args.config, args.weight, args.output)


if __name__ == "__main__":
    main()
