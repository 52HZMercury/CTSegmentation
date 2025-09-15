import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def validation(Trainer, epoch_iterator_val):
    Trainer.model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device))
            # 将image和all_lab在通道维度上拼接
            all_lab = batch["all_lab"].to(Trainer.device)
            val_inputs = torch.cat((val_inputs, all_lab), dim=1)

            val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, Trainer.model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [Trainer.post_label(val_labels_tensor) for val_labels_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_outputs_convert = [Trainer.post_pred(val_outputs_tensor) for val_outputs_tensor in val_outputs_list]
            Trainer.dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validating")
        mean_dice_val = Trainer.dice_metric.aggregate().item()
        Trainer.dice_metric.reset()
    return mean_dice_val