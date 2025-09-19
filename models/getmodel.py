"""
Model initialization module based on config file
"""
import yaml
import torch
from monai.networks.nets import BasicUnet
from models.segmamba import SegMamba

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
    elif architecture == 'SegMamba':
        model = SegMamba(in_chans=config['model']['in_channels'],
                         out_chans=config['model']['out_channels'],
                         depths=[2, 2, 2, 2],
                         feat_size=[16, 32, 64, 128])

    elif architecture == 'EchoMamba':
        model = SegMamba(in_chans=config['model']['in_channels'],
                         out_chans=config['model']['out_channels'],
                         depths=[2, 2, 2, 2],
                         feat_size=[16, 32, 64, 128])
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model
