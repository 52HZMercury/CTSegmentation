"""
使用epoch控制训练过程的主训练文件
"""
import os
import tempfile
import yaml
import torch
from trainer import Trainer

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
    current_epoch = 0
    dice_val_best = 0.0
    global_step_best = 0

    print(f"开始训练，总共 {max_epochs} 个epochs")
    print(f"模型将保存到: {root_dir}")

    # 训练循环
    while current_epoch < max_epochs:
        current_epoch += 1
        current_epoch, dice_val_best, global_step_best = train(trainer, current_epoch, dice_val_best, global_step_best)

    print(f"训练完成! 最佳Dice系数: {dice_val_best}")


def train(Trainer, current_epoch, dice_val_best, global_step_best):
    Trainer.model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(Trainer.train_loader)

    # 返回一个迭代器，并会不断打印迭代进度条，desc为进度条前缀，允许调整窗口大小
    from tqdm import tqdm
    epoch_iterator = tqdm(Trainer.train_loader, desc=f"Epoch {current_epoch} Training", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        # 读取图像和标签并放入GPU
        x, y = (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device))
        # 将图像放入模型进行训练
        logit_map = Trainer.model(x)
        # 计算训练损失
        loss = Trainer.loss_function(logit_map, y)
        # 向后传播
        loss.backward()
        # 计算损失和
        epoch_loss += loss.item()
        # 更新参数
        Trainer.optimizer.step()
        # 更新参数之后清除梯度
        Trainer.optimizer.zero_grad()
        # 设置进度条前缀，为当前训练次数，训练总数，本次训练损失
        epoch_iterator.set_description(f"Epoch {current_epoch} Training (loss=%2.5f)" % loss)

        # 每个epoch结束后进行验证
        if step == num_steps:
            from validation import validation
            epoch_iterator_val = tqdm(Trainer.val_loader, desc=f"Epoch {current_epoch} Validation", dynamic_ncols=True)
            dice_val = validation(Trainer, epoch_iterator_val)
            epoch_loss /= step
            Trainer.epoch_loss_values.append(epoch_loss)
            Trainer.metric_values.append(dice_val)
            Trainer.scheduler.step(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = current_epoch
                torch.save(Trainer.model.state_dict(),
                           os.path.join(config['data']['out_dir'], f"best_metric_model_{dice_val}.pth"))
                print(f'Saved! Current best average dice:{dice_val_best}')
            else:
                print(f'Not saved! Current best average dice:{dice_val_best}, current average dice:{dice_val}')

    # 返回当前epoch，最佳Dice系数，最佳epoch
    return current_epoch, dice_val_best, global_step_best


if __name__ == "__main__":
    main()
