import yaml
import torch
import os
from tqdm import tqdm
from validation import validation

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def train(Trainer, current_epoch, dice_val_test, clDice_val_test, global_step_best):
    Trainer.model.train()
    epoch_loss = 0
    step = 0
    num_steps = len(Trainer.train_loader)

    # 返回一个迭代器，并会不断打印迭代进度条，desc为进度条前缀，允许调整窗口大小
    epoch_iterator = tqdm(Trainer.train_loader, desc=f"{config['data']['exp_name']} Epoch {current_epoch} Training", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        # 读取图像和标签并放入GPU
        x, y, all_lab= (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device), batch["all_lab"].to(Trainer.device))

        # 在通道维度上拼接x和all_lab
        x = torch.cat((x, all_lab), dim=1)

        # x, y = (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device).to(Trainer.device))

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
        epoch_iterator.set_description(f"{config['data']['exp_name']} Epoch {current_epoch} Training (loss=%2.5f)" % loss)

        # 记录每个step的训练损失到TensorBoard
        Trainer.writer.add_scalar('Train/Loss', loss.item(), Trainer.global_step)
        Trainer.global_step += 1

        # 每个epoch结束后进行验证
        if step == num_steps:
            epoch_iterator_val = tqdm(Trainer.val_loader, desc=f"Epoch {current_epoch} Validation", dynamic_ncols=True)

            # [修改] 接收两个返回值
            dice_val, cldice_val = validation(Trainer, epoch_iterator_val)

            epoch_loss /= step
            Trainer.epoch_loss_values.append(epoch_loss)
            Trainer.metric_values.append(dice_val)
            Trainer.scheduler.step(dice_val)

            # [修改] 记录 clDice 到 TensorBoard
            Trainer.writer.add_scalar('Train/Epoch_Average_Loss', epoch_loss, current_epoch)
            Trainer.writer.add_scalar('Validation/Dice', dice_val, current_epoch)
            Trainer.writer.add_scalar('Validation/clDice', cldice_val, current_epoch)  # 新增

            if dice_val > dice_val_test:
                dice_val_test = dice_val
                global_step_best = current_epoch
                clDice_val_test = cldice_val
                # 确保checkpoint目录存在
                checkpoint_dir = os.path.join(config['data']['out_dir'], f"{config['data']['exp_name']}/checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                torch.save(Trainer.model.state_dict(),
                           os.path.join(checkpoint_dir, f"best_metric_model_{dice_val:.4f}.pth"))
                # [修改] 打印信息增加 clDice
                print(f'Saved! Best Dice:{dice_val_test:.4f}, clDice: {clDice_val_test:.4f}')
                Trainer.writer.add_scalar('Validation/Best_Dice', dice_val_test, current_epoch)

            else:
                # [修改] 打印信息增加 clDice
                print(
                    f'Not saved! Best Dice:{dice_val_test:.4f}, Cur Dice:{dice_val:.4f}, Cur clDice: {cldice_val:.4f}')

    return current_epoch, dice_val_test, clDice_val_test, global_step_best
