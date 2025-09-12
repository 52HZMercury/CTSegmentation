import yaml
import torch
import os
from tqdm import tqdm
from validation import validation

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# def train(0, train_loader, 0.0, 0)
def train(Trainer, current_epoch, dice_val_best, global_step_best):
    Trainer.model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(Trainer.train_loader)

    # 返回一个迭代器，并会不断打印迭代进度条，desc为进度条前缀，允许调整窗口大小
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
        # .item()返回指定位置的高精度值
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
