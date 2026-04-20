import yaml
import torch
import os
import glob  # 新增：用于查找旧的权重文件
from tqdm import tqdm
from validation import validation



def train(Trainer, current_epoch, dice_val_test, iou_val_test, clDice_val_test, global_step_best, early_stop_counter, config):
    """
    Args:
        early_stop_counter (int): 当前性能未提升的连续轮数
    Returns:
        tuple: 包含更新后的各项指标及 early_stop_counter，以及是否触发停止的标志
    """
    Trainer.model.train()
    epoch_loss = 0
    num_steps = len(Trainer.train_loader)
    stop_training = False  # 是否停止训练的标志

    # 设定早停耐心值
    patience = config['training']['patience']

    epoch_iterator = tqdm(Trainer.train_loader, desc=f"{config['data']['exp_name']} Epoch {current_epoch} Training",
                          dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):

        step += 1
        x, y = (batch["image"].to(Trainer.device), batch["label"].to(Trainer.device))
        # x, all_lab, y = (batch["image"].to(Trainer.device), batch["all_lab"].to(Trainer.device), batch["label"].to(Trainer.device))
        # x = torch.cat((x, all_lab), dim=1)

        logit_map = Trainer.model(x)
        loss = Trainer.loss_function(logit_map, y)

        loss.backward()
        epoch_loss += loss.item()
        Trainer.optimizer.step()
        Trainer.optimizer.zero_grad()

        epoch_iterator.set_description(
            f"{config['data']['exp_name']} Epoch {current_epoch} Training (loss=%2.5f)" % loss)
        Trainer.writer.add_scalar('Train/Loss', loss.item(), Trainer.global_step)
        Trainer.global_step += 1

        # 每个epoch结束后进行验证
        if step == num_steps:
            epoch_iterator_val = tqdm(Trainer.val_loader, desc=f"Epoch {current_epoch} Validation", dynamic_ncols=True)
            dice_val, iou_val, cldice_val = validation(Trainer, epoch_iterator_val)

            epoch_loss /= step
            Trainer.epoch_loss_values.append(epoch_loss)
            Trainer.metric_values.append(dice_val)
            Trainer.scheduler.step(dice_val)

            Trainer.writer.add_scalar('Train/Epoch_Average_Loss', epoch_loss, current_epoch)
            Trainer.writer.add_scalar('Validation/Dice', dice_val, current_epoch)
            Trainer.writer.add_scalar('Validation/IoU', iou_val, current_epoch)
            Trainer.writer.add_scalar('Validation/clDice', cldice_val, current_epoch)

            # 核心逻辑：早停判断
            if dice_val > dice_val_test:
                # 性能提升，重置计数器
                dice_val_test = dice_val
                iou_val_test = iou_val
                clDice_val_test = cldice_val
                global_step_best = current_epoch
                early_stop_counter = 0

                # 保存模型
                checkpoint_dir = os.path.join(config['data']['out_dir'], f"{config['data']['exp_name']}/checkpoint")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                else:
                    # 【修改点】查找并删除该目录下以前保存的 best_metric_model 权重
                    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "best_metric_model_*.pth"))
                    for old_ckpt in old_checkpoints:
                        try:
                            os.remove(old_ckpt)
                        except OSError:
                            pass

                # 保存新的权重文件
                torch.save(Trainer.model.state_dict(),
                           os.path.join(checkpoint_dir, f"best_metric_model_{dice_val:.4f}.pth"))

                print(f'Saved! Best Dice:{dice_val_test:.4f}, IoU: {iou_val_test:.4f}, clDice: {clDice_val_test:.4f}')
                Trainer.writer.add_scalar('Validation/Best_Dice', dice_val_test, current_epoch)
            else:
                # 性能未提升，计数器累加
                early_stop_counter += 1
                print(
                    f'Not saved! Best Dice:{dice_val_test:.4f}, Cur Dice:{dice_val:.4f}, EarlyStop: {early_stop_counter}/{patience}')

            # 检查是否达到阈值
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {current_epoch} epochs.")
                stop_training = True

    return current_epoch, dice_val_test, iou_val_test, clDice_val_test, global_step_best, early_stop_counter, stop_training