"""
使用epoch控制训练过程的主训练文件
"""
import os
import tempfile
import yaml
from train import train
from trainer import Trainer
from datetime import datetime  # 新增：引入 datetime

import warnings  # 添加这一行
warnings.filterwarnings("ignore")
# 加载配置
config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# ==========================================
# 动态生成实验名称: exp_YYMMDD-HHMM
# ==========================================
current_time = datetime.now().strftime("%y%m%d-%H%M")
config['data']['exp_name'] = f"exp_{current_time}"
# ==========================================


def main():
    # 创建训练器实例
    trainer = Trainer(config=config)

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
        (current_epoch,
         dice_val_test,
         iou_val_test,
         clDice_val_test,
         global_step_best,
         early_stop_counter,
         stop_now) = train(
            trainer,
            current_epoch,
            dice_val_test,
            iou_val_test,
            clDice_val_test,
            global_step_best,
            early_stop_counter,
            config
        )
        if stop_now:
            break

    print("")
    print(f"=======================训练完成========================")
    print(f"== 实验名称  : {config['data']['exp_name']}")
    print(f"== 实验模型  : {config['model']['architecture']}")
    print(f"== test_dice: {dice_val_test}")
    print(f"== test_iou : {iou_val_test:.4f}")
    print(f"== test_clDice: {clDice_val_test}")
    print(f"=======================================================")



if __name__ == "__main__":
    main()
