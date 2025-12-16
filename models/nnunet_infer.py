#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   nnunet_infer.py
@Time    :   2025/05/21 14:33:37
@Author  :   Hui Sun
@Version :   1.0
@Contact :   bitsunhui@163.com
@License :   (C)Copyright 2018-2025, Hui Sun
@Desc    :   None
"""

import os
import warnings

# set nnUNet environment variables
tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
os.environ["nnUNet_raw"] = os.path.join(tmp_path, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(tmp_path, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(tmp_path, "nnUNet_results")

import nnunetv2
import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule

warnings.filterwarnings("ignore")


class CustomnnUNetPredictor(nnUNetPredictor):
    def initialize_from_dir(
        self,
        dataset_json_dir: str,
        plan_json_dir: str,
        checkpoint_dir: str,
    ):
        """
        This is used when making predictions with a trained model
        """

        dataset_json = load_json(dataset_json_dir)
        plans = load_json(plan_json_dir)
        plans_manager = PlansManager(plans)

        parameters = []
        checkpoint = torch.load(
            checkpoint_dir,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        trainer_name = checkpoint["trainer_name"]
        configuration_name = checkpoint["init_args"]["configuration"]
        inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else None
        )
        parameters.append(checkpoint["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            "nnunetv2.training.nnUNetTrainer",
        )
        if trainer_class is None:
            raise RuntimeError(
                "Unable to locate trainer class {} in nnunetv2.training.nnUNetTrainer. ".format(
                    trainer_name
                ),
                "Please place it there (in any .py file)!",
            )
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if (
            ("nnUNet_compile" in os.environ.keys())
            and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t"))
            and not isinstance(self.network, OptimizedModule)
        ):
            print("Using torch.compile")
            self.network = torch.compile(self.network)


if __name__ == "__main__":
    # set weights path
    model_path = "./model_weights"
    dataset_json_dir = os.path.join(model_path, "dataset.json")
    plan_json_dir = os.path.join(model_path, "plans.json")
    checkpoint_dir = os.path.join(model_path, "nnunet_checkpoint_best.pth")

    # init the predictor
    predictor = CustomnnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_dir(
        dataset_json_dir=dataset_json_dir,
        plan_json_dir=plan_json_dir,
        checkpoint_dir=checkpoint_dir,
    )

    # predict a single image
    input_img_dir = "./test_data/image.nii.gz"
    save_mask_dir = "./test_data/pred_mask.nii.gz"

    img, props = SimpleITKIO().read_images([input_img_dir])
    pred_mask = predictor.predict_single_npy_array(
        input_image=img,
        image_properties=props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )

    pred_simg = sitk.GetImageFromArray(pred_mask)  # type: ignore
    pred_simg.SetSpacing(props["sitk_stuff"]["spacing"])
    pred_simg.SetOrigin(props["sitk_stuff"]["origin"])
    pred_simg.SetDirection(props["sitk_stuff"]["direction"])

    sitk.WriteImage(pred_simg, save_mask_dir)
    print("done")
