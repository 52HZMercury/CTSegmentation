import torch
import yaml
import numpy as np
from skimage.morphology import skeletonize
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def compute_cl_dice(pred, target):
    """
    计算 clDice (Centerline Dice)
    """
    # 确保输入是布尔类型，skeletonize 需要布尔或二值输入
    pred = (pred > 0.5)
    target = (target > 0.5)

    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 1.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return 0.0

    # 提取骨架 (在 CPU 上运行)
    tprec_skel = skeletonize(pred)
    tsens_skel = skeletonize(target)

    # 计算 Topology Precision (Tprec)
    tprec = np.sum(tprec_skel * target) / (np.sum(tprec_skel) + 1e-5)

    # 计算 Topology Sensitivity (Tsens)
    tsens = np.sum(tsens_skel * pred) / (np.sum(tsens_skel) + 1e-5)

    # 计算调和平均数
    if tprec + tsens == 0:
        return 0.0

    return 2.0 * tprec * tsens / (tprec + tsens)


def validation(Trainer, epoch_iterator_val):
    Trainer.model.eval()
    Trainer.dice_metric.reset()
    Trainer.cldice_metric.reset()

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

            # 计算常规 Dice
            Trainer.dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

            # 计算 clDice
            for pred_tensor, label_tensor in zip(val_outputs_convert, val_labels_convert):
                # 转为 numpy
                pred_np = pred_tensor.cpu().numpy()
                label_np = label_tensor.cpu().numpy()

                class_cldices = []
                num_classes = pred_np.shape[0]

                # 遍历前景类 (跳过背景类 0)
                for c in range(1, num_classes):
                    score = compute_cl_dice(pred_np[c], label_np[c])
                    class_cldices.append(score)

                if class_cldices:
                    # [修复] 使用 append 添加数据，并转为 Tensor
                    mean_val = np.mean(class_cldices)
                    Trainer.cldice_metric.append(torch.tensor(mean_val, device=Trainer.device))

            epoch_iterator_val.set_description("Validating")

        # 获取最终结果
        mean_dice_val = Trainer.dice_metric.aggregate().item()

        # [修复] 此时 cldice_metric 中包含 Tensor，可以安全调用 .item()
        # 如果验证集为空导致没有添加任何数据，这里可能会报错，添加简单的异常处理更稳健
        try:
            mean_cldice_val = Trainer.cldice_metric.aggregate().item()
        except AttributeError:
            # 如果返回的是普通数值（如0），直接使用
            mean_cldice_val = Trainer.cldice_metric.aggregate()

        Trainer.dice_metric.reset()
        Trainer.cldice_metric.reset()

    return mean_dice_val, mean_cldice_val