"""
Model initialization module based on config file
"""
import yaml
import torch
from monai.networks.nets import BasicUnet

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def create_model():
    """
    Create model based on config parameters

    Returns:
        torch.nn.Module: Initialized model
    """
    architecture = config['model'].get('architecture', 'BasicUNet')

    if architecture == 'BasicUNet':
        model = BasicUnet(
            spatial_dims=config['model']['spatial_dims'],
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model
