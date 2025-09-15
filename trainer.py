import yaml
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from data.Augmentation import train_transforms, val_transforms
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
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
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    """
    训练器类，用于初始化和管理训练过程中的各种参数和组件
    """

    def __init__(self, config_path="config/config.yaml"):
        """
        初始化训练器

        Args:
            config_path (str): 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化设备
        self.device = self._init_device()
        
        # 初始化数据集和数据加载器
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # 初始化模型
        self.model = self._init_model()
        self.model.to(self.device)
        
        # 初始化损失函数
        self.loss_function = self._init_loss_function()
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._init_scheduler()

        # 设置TensorBoard日志写入器
        log_dir = os.path.join(self.config['data']['out_dir'], self.config['data']['exp_name'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 设置其他训练参数
        # self.max_iterations = self.config['training']['max_iterations']
        # self.eval_num = self.config['training']['eval_num']
        self.post_label = AsDiscrete(to_onehot=self.config['training']['num_class'])
        # 采用argmax函数
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.config['training']['num_class'])
        # 计算两个张量之间的平均Dice系数
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        # 全局训练次数
        self.global_step = 0
        self.epoch_loss_values = []
        self.metric_values = []

        
        # 为每个卷积层寻找适合的卷积实现算法，加速网络训练
        torch.backends.cudnn.benchmark = True

    def _init_device(self):
        """
        初始化设备 (CPU/GPU)
        
        Returns:
            torch.device: 设备对象
        """
        device_config = self.config.get('device', {})
        cuda_device = device_config.get('cuda_device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(cuda_device)

    def _init_data_loaders(self):
        """
        初始化数据集和数据加载器
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        # 数据
        split_json = self.config['data']['split_json']
        datasets = split_json
        
        # 训练集
        datalist = load_decathlon_datalist(
            datasets, 
            is_segmentation=True, 
            data_list_key=self.config['data']['datasets_key']
        )
        val_files = load_decathlon_datalist(
            datasets, 
            is_segmentation=True, 
            data_list_key=self.config['data']['validation_key']
        )
        
        # 具有缓存机制的dataset
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=self.config['training']['cache_num'],
            cache_rate=self.config['training']['cache_rate'],
            num_workers=self.config['training']['num_workers'],
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=self.config['validation']['cache_num'],
            cache_rate=self.config['validation']['cache_rate'],
            num_workers=self.config['validation']['num_workers'],
        )
        
        # 数据加载器
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True, 
            num_workers=self.config['training']['num_workers'], 
            pin_memory=self.config['training']['pin_memory']
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config['validation']['batch_size'], 
            shuffle=False, 
            num_workers=self.config['validation']['num_workers'], 
            pin_memory=self.config['validation']['pin_memory']
        )
        
        return train_loader, val_loader

    def _init_model(self):
        """
        初始化模型
        
        Returns:
            torch.nn.Module: 模型对象
        """
        from models.getmodel import create_model
        return create_model()

    def _init_loss_function(self):
        """
        初始化损失函数
        
        Returns:
            torch.nn.Module: 损失函数对象
        """
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def _init_optimizer(self):
        """
        初始化优化器
        
        Returns:
            torch.optim.Optimizer: 优化器对象
        """
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'], 
            weight_decay=self.config['training']['weight_decay']
        )

    def _init_scheduler(self):
        """
        初始化学习率调度器
        
        Returns:
            torch.optim.lr_scheduler: 学习率调度器对象
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'max',
            factor=float(self.config['scheduler']['factor']),
            verbose=True,
            min_lr=float(self.config['scheduler']['min_lr']),
            patience=int(self.config['scheduler']['patience'])
        )
