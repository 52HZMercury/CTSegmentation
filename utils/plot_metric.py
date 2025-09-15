import torch
import yaml
import matplotlib.pyplot as plt

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
def plot_loss_and_metric(Trainer):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [config['training']['eval_num'] * (i + 1) for i in range(len(Trainer.epoch_loss_values))]
    y = Trainer.epoch_loss_values
    plot_data_1 = [x, y]
    torch.save(plot_data_1, '/workdir2/cn24/program/CT_SU/logs/plot_loss.pth')
    plt.xlabel("Iteration")
    plt.plot(plot_data_1[0], plot_data_1[1])
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [config['training']['eval_num'] * (i + 1) for i in range(len(Trainer.metric_values))]
    y = Trainer.metric_values
    plot_data_2 = [x, y]
    torch.save(plot_data_2, '/workdir2/cn24/program/CT_SU/logs/plot_dice.pth')
    plt.xlabel("Iteration")
    plt.plot(plot_data_2[0], plot_data_2[1])
    plt.show()