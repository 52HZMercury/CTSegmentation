"""
Model initialization module based on config file
"""
import os
import sys
import yaml
import inspect  # 新增: 用于动态检查函数签名
import torch
import pydoc  # <--- 新增导入：用于动态加载类
from monai.networks.nets import BasicUnet
from models.segmamba import SegMamba

# --- 1. 设置临时的 nnU-Net 环境变量以消除警告 ---
tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path, exist_ok=True)
os.environ["nnUNet_raw"] = os.path.join(tmp_path, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(tmp_path, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(tmp_path, "nnUNet_results")
# -------------------------------------------------------------------------

# --- Numpy 兼容性处理 ---
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
# ---------------------------

# --- nnU-Net 依赖库 ---
import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
# ----------------------------



# import os
# import sys
# import yaml
# import inspect
# import torch
# import pydoc  # <--- 新增导入：用于动态加载类
# from monai.networks.nets import BasicUnet
# from models.segmamba import SegMamba
#
# # ... (保持原本的环境变量设置和 Numpy 兼容性代码不变) ...
#
# # ... (保持原本的 import nnunetv2 相关代码不变) ...
# import nnunetv2
# from batchgenerators.utilities.file_and_folder_operations import join, load_json
# from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
# from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
# from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
#


config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_nnunet_model(checkpoint_path, plans_path=None, dataset_json_path=None):
    """
    动态构建并加载 nnU-Net 模型，支持显式架构配置
    """
    # 自动推断路径
    model_dir = os.path.dirname(checkpoint_path)
    if plans_path is None:
        plans_path = os.path.join(model_dir, "plans.json")
    if dataset_json_path is None:
        dataset_json_path = os.path.join(model_dir, "dataset.json")

    if not os.path.exists(plans_path) or not os.path.exists(dataset_json_path):
        raise FileNotFoundError(
            f"nnU-Net requires 'plans.json' and 'dataset.json' in {model_dir}"
        )

    # 加载配置
    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_path)
    plans_manager = PlansManager(plans)

    # 加载 Checkpoint 元数据
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 检查checkpoint格式并获取configuration_name
    if "init_args" in checkpoint:
        configuration_name = checkpoint["init_args"]["configuration"]
    elif "configuration" in checkpoint:
        configuration_name = checkpoint["configuration"]
    else:
        print("Warning: checkpoint does not contain configuration info. Using default '3d_fullres'")
        configuration_name = "3d_fullres"

    configuration_manager = plans_manager.get_configuration(configuration_name)

    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    num_segmentation_heads = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

    # 获取配置字典
    config_dict = configuration_manager.configuration

    # -------------------------------------------------------
    # 核心修复: 判断是否使用显式架构 (Architecture Dict)
    # -------------------------------------------------------
    if 'architecture' in config_dict:
        print("Detected explicit architecture configuration. Building manually...")

        # 1. 提取架构信息
        arch_config = config_dict['architecture']
        network_class_name = arch_config['network_class_name']
        arch_kwargs = arch_config['arch_kwargs']
        arch_kwargs_req_import = arch_config.get('_kw_requires_import', [])

        # 2. 动态解析需要 import 的参数 (例如 conv_op, norm_op)
        for k in arch_kwargs_req_import:
            if k in arch_kwargs and isinstance(arch_kwargs[k], str):
                arch_kwargs[k] = pydoc.locate(arch_kwargs[k])

        # 3. 获取网络类
        network_class = pydoc.locate(network_class_name)
        if network_class is None:
            raise ImportError(f"Could not locate network class: {network_class_name}")

        # 4. 补充输入输出通道参数
        kw_to_pass = arch_kwargs.copy()
        kw_to_pass['input_channels'] = num_input_channels
        kw_to_pass['num_classes'] = num_segmentation_heads

        # 5. 实例化网络
        network = network_class(**kw_to_pass)

        # 6. 处理深监督 (Deep Supervision)
        if hasattr(network, 'decoder') and hasattr(network.decoder, 'deep_supervision'):
            network.decoder.deep_supervision = False

    else:
        # --- 传统路径: 使用 Trainer 构建 (适用于旧版/标准 UNet) ---
        print("Using Trainer to build network architecture...")

        # 检查trainer_name是否存在
        if "trainer_name" in checkpoint:
            trainer_name = checkpoint["trainer_name"]
        else:
            print("Warning: checkpoint does not contain trainer_name. Using default 'nnUNetTrainer'")
            trainer_name = "nnUNetTrainer"

        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            "nnunetv2.training.nnUNetTrainer",
        )
        if trainer_class is None:
            raise RuntimeError(f"Could not find trainer class {trainer_name}")

        trainer = trainer_class(
            plans=plans,
            configuration=configuration_name,
            fold=0,
            dataset_json=dataset_json,
            unpack_dataset=False,
            device=torch.device('cpu')
        )

        possible_args = {
            "plans_manager": plans_manager,
            "dataset_json": dataset_json,
            "configuration_manager": configuration_manager,
            "num_input_channels": num_input_channels,
            "num_output_channels": num_segmentation_heads,
            "enable_deep_supervision": False
        }

        build_method = trainer.build_network_architecture
        sig = inspect.signature(build_method)
        call_kwargs = {}
        for param_name in sig.parameters:
            if param_name == 'self': continue
            if param_name in possible_args:
                call_kwargs[param_name] = possible_args[param_name]
            elif param_name == 'num_classes' and 'num_output_channels' in possible_args:
                call_kwargs[param_name] = possible_args['num_output_channels']

        network = build_method(**call_kwargs)

    # -------------------------------------------------------

    # 加载权重
    print("Loading network weights...")

    # 处理不同格式的权重键
    if "network_weights" in checkpoint:
        state_dict = checkpoint["network_weights"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        print("Warning: Could not find standard weight keys. Assuming checkpoint itself is state_dict")
        state_dict = checkpoint

    network.load_state_dict(state_dict)

    return network


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

    elif architecture == 'nnUNet':
        checkpoint_path = config['model']['checkpoint_path']
        print(f"Loading nnU-Net from: {checkpoint_path}")
        model = load_nnunet_model(checkpoint_path)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model