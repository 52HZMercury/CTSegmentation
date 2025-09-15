import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
import yaml
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
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

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUnet
from data.Augmentation import train_transforms, val_transforms
from models.getmodel import create_model
from utils.plot_metric import plot_loss_and_metric
from train import train
from trainer import Trainer
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
warnings.filterwarnings("ignore")


# Load configuration
config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 设置数据路径
directory = config['data']['out_dir']
out_dir = tempfile.mkdtemp() if directory is None else directory


# 数据
# data_dir = "/workdir2/cn24/data/Takayasu_rename/"
split_json = config['data']['split_json']
datasets = split_json
# 从json文件中加载数据集（数据集路径，是否用于分割任务，字典键值）
# 训练集
datalist = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key=config['data']['datasets_key'])
val_files = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key=config['data']['validation_key'])
# 具有缓存机制的dataset，在每一个epoch训练之前，把训练的数据加载进缓存
# 读取图像数并进行图像转换
# （将image和label的地址或值存为字典，transform，要缓存的项目数，，缓存数据占总数的百分比默认1，要使用的工作进程数）
# train_ds = CacheDataset(
#     data=datalist,
#     transform=train_transforms,
#     cache_num=config['training']['cache_num'],
#     cache_rate=config['training']['cache_rate'],
#     num_workers=config['training']['num_workers'],
# )
# val_ds = CacheDataset(
#     data=val_files,
#     transform=val_transforms,
#     cache_num=config['validation']['cache_num'],
#     cache_rate=config['validation']['cache_rate'],
#     num_workers=config['validation']['num_workers'],
# )
# # 加载图像，（加载的数据集，batch_size，是否打乱，使用的进程数，是否将数据保存在pin_memory区）
# train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=config['training']['pin_memory'])
# val_loader = DataLoader(val_ds, batch_size=config['validation']['batch_size'], shuffle=False, num_workers=config['validation']['num_workers'], pin_memory=config['validation']['pin_memory'])
#
# # 设置设备
# device_config = config.get('device', {})
# cuda_device = device_config.get('cuda_device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device(cuda_device)
#
# # 设置网络
# model = create_model()
# model.to(device)
#
# # 损失函数
# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# # 为每个卷积层寻找适合的卷积实现算法，加速网络训练
# torch.backends.cudnn.benchmark = True
# # 设置网络参数更新方式
# optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config['scheduler']['factor'], verbose=1, min_lr=config['scheduler']['min_lr'],
#                                                        patience=config['scheduler']['patience'])

Trainer(config_path)

# def validation(epoch_iterator_val):
#     model.eval()
#     with torch.no_grad():
#         for batch in epoch_iterator_val:
#             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#             val_outputs = sliding_window_inference(val_inputs, (64, 64, 64), 4, model)
#             val_labels_list = decollate_batch(val_labels)
#             val_labels_convert = [post_label(val_labels_tensor) for val_labels_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(val_outputs)
#             val_outputs_convert = [post_pred(val_outputs_tensor) for val_outputs_tensor in val_outputs_list]
#             dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
#             epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, max_iterations))
#         mean_dice_val = dice_metric.aggregate().item()
#         dice_metric.reset()
#     return mean_dice_val





# def plot_loss_and_metric():
#     plt.figure("train", (12, 6))
#     plt.subplot(1, 2, 1)
#     plt.title("Iteration Average Loss")
#     x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
#     y = epoch_loss_values
#     plot_data_1 = [x, y]
#     torch.save(plot_data_1, '/workdir2/cn24/program/CT_SU/logs/plot_loss.pth')
#     plt.xlabel("Iteration")
#     plt.plot(plot_data_1[0], plot_data_1[1])
#     plt.subplot(1, 2, 2)
#     plt.title("Val Mean Dice")
#     x = [eval_num * (i + 1) for i in range(len(metric_values))]
#     y = metric_values
#     plot_data_2 = [x, y]
#     torch.save(plot_data_2, '/workdir2/cn24/program/CT_SU/logs/plot_dice.pth')
#     plt.xlabel("Iteration")
#     plt.plot(plot_data_2[0], plot_data_2[1])
#     plt.show()


# max_iterations = config['training']['max_iterations']
# max_iterations = 3400
# eval_num = config['training']['eval_num']

# 将输入的张量转换为离散值，采2位独热编码
post_label = AsDiscrete(to_onehot=2)
# 采用argmax函数
post_pred = AsDiscrete(argmax=True, to_onehot=2)
# 计算两个张量之间的平均Dice系数
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# 全局训练次数
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_3_0.8936.pth")))
# 当训练iteration小于max_iterations，继续训练

if __name__ == '__main__':
    while global_step < config['training']['max_iterations']:
        global_step, dice_val_best, global_step_best = train(Trainer, global_step, dice_val_best, global_step_best)
    plot_loss_and_metric(Trainer)
