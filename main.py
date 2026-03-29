"""
使用epoch控制训练过程的主训练文件
"""
import os
import tempfile
import yaml
from train import train
from trainer import Trainer

import warnings  # 添加这一行
warnings.filterwarnings("ignore")
# 加载配置
config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def main():
    # 创建训练器实例
    trainer = Trainer(config_path)

    # 设置输出目录
    out_dir = config['data']['out_dir']
    root_dir = tempfile.mkdtemp() if out_dir is None else out_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # 训练参数
    max_epochs = config['training']['max_epochs']
    early_stop_counter = 0
    current_epoch = 0
    dice_val_test = 0.0
    iou_val_test = 0.0
    clDice_val_test = 0.0
    global_step_best = 0

    print(f"开始训练---------------------------------------------------------")

    # 训练循环
    for epoch in range(max_epochs):
        current_epoch += 1
        current_epoch, dice_val_test, iou_val_test, clDice_val_test, global_step_best, early_stop_counter, stop_now = train(trainer, current_epoch, dice_val_test, iou_val_test, clDice_val_test, global_step_best, early_stop_counter)
        if stop_now:
            break

    print("")
    print(f"=======================训练完成========================")
    print(f"== exp_name : {config['data']['exp_name']}")
    print(f"== test_dice: {dice_val_test}")
    print(f"== test_iou : {iou_val_test:.4f}")
    print(f"== test_clDice: {clDice_val_test}")
    print(f"=======================================================")



if __name__ == "__main__":
    main()
