import os
import copy
import math
from typing import Dict, Any

import torch
from torch import nn
from torch.nn import Parameter
import torch.utils.checkpoint as checkpoint

import torchvision
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import wordpunct_tokenize
from einops import rearrange
from peft import get_peft_model, LoraConfig
from huggingface_hub import hf_hub_download

__all__ = ["Merlin"]

# ==============================================================================
# 1. UTILS (merlin/utils.py & build.py sanitization)
# ==============================================================================

def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path

def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))


# ==============================================================================
# 2. INFLATE (merlin/models/inflate.py)
# ==============================================================================

def inflate_conv(
    conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False
):
    if conv2d.kernel_size[0] == 7:
        kernel_dim = (3, 7, 7)
        padding = (1, 3, 3)
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    else:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim, linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode,
            )
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError(
                "{} is not among known pooling classes".format(type(pool2d))
            )

    return pool3d


# ==============================================================================
# 3. I3RESNET (merlin/models/i3res.py)
# ==============================================================================

class I3ResNet(torch.nn.Module):
    def __init__(
        self,
        resnet2d,
        frame_nb=16,
        class_nb=1000,
        conv_class=False,
        return_skips=False,
        ImageEmbedding=False,
        PhenotypeCls=False,
        FiveYearPred=False,
    ):
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        self.conv1 = inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True,
            )
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim
            )
            self.fc = inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x, use_reentrant=False)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        if self.conv_class:
            x_features = self.avgpool(x)

            if self.ImageEmbedding:
                return x_features.squeeze(2).squeeze(2).squeeze(2).unsqueeze(0)

            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                if self.PhenotypeCls or self.FiveYearPred:
                    probs = torch.sigmoid(x_ehr)
                    return probs

                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out

        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if x.requires_grad:
            out = checkpoint.checkpoint(run_function, x, use_reentrant=False)
        else:
            out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d


# ==============================================================================
# 4. BUILD / ARCHITECTURE (merlin/models/build.py)
# ==============================================================================

class ImageEncoder(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        FiveYearPred: bool = False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692 if not self.FiveYearPred else 6,
            conv_class=True,
            ImageEmbedding=self.ImageEmbedding,
            PhenotypeCls=self.PhenotypeCls,
            FiveYearPred=self.FiveYearPred,
        )

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        elif self.PhenotypeCls:
            return self.i3_resnet(image)
        elif self.FiveYearPred:
            return self.i3_resnet(image)
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(
        self,
        init_logit_scale: float = 1.0,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        FiveYearPred: bool = False,
    ):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.PhenotypeCls = PhenotypeCls
        self.FiveYearPred = FiveYearPred
        self.encode_image = ImageEncoder(
            ImageEmbedding=self.ImageEmbedding,
            PhenotypeCls=self.PhenotypeCls,
            FiveYearPred=self.FiveYearPred,
        )
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            return image_features
        elif self.PhenotypeCls and text is None:
            phenotype_features = self.encode_image(image)
            return phenotype_features
        elif self.FiveYearPred and text is None:
            five_year_features = self.encode_image(image)
            return five_year_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif self.PhenotypeCls and text is not None:
            raise ValueError("Text input not required for phenotype classification")
        elif self.FiveYearPred and text is not None:
            raise ValueError("Text input not required for five year disease prediction")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")

        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )


# ==============================================================================
# 5. RADIOLOGY REPORT GENERATION (merlin/models/radiology_report_generation.py)
# ==============================================================================

class ModifiedI3ResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedI3ResNet, self).__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
        )

    def forward(self, x):
        x = self.features(x)
        # flatten the outputs torch.Size([8, 2048, 10, 7, 7]) -> torch.Size([8, 2048, 490])
        x = torch.flatten(x, start_dim=2)
        # swap 1st and 2nd dimensions torch.Size([8, 2048, 490]) -> torch.Size([8, 490, 2048])
        x = rearrange(x, "b c n -> b n c")
        return x


class ModifiedImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the i3d resnet model
        resnet = torchvision.models.resnet152(pretrained=True)
        i3_resnet = I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692,
            conv_class=True,
            ImageEmbedding=True,
            PhenotypeCls=False,
        )

        # Modify the i3d resnet model
        modified_model = ModifiedI3ResNet(i3_resnet)

        self.vision_model = modified_model

    def forward(self, images):
        images = images.permute(0, 1, 4, 2, 3)
        images = torch.cat((images, images, images), dim=1)
        image_embeddings = self.vision_model(images)
        if len(image_embeddings.shape) == 2:
            image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings


class Adapter(nn.Module):
    def __init__(self, input_size, output_size):
        super(Adapter, self).__init__()
        self.linear2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear2(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "StanfordAIMI/RadLLaMA-7b",
            use_fast=True,
            trust_remote_code=True,
            cache_dir="./checkpoints",
        )
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/RadLLaMA-7b",
            cache_dir="./checkpoints",
        )
        self.text_decoder.gradient_checkpointing_enable()

        peft_params = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=512, bias="none", task_type="CAUSAL_LM"
        )

        self.text_decoder = get_peft_model(self.text_decoder, peft_params)
        self.text_decoder.print_trainable_parameters()

    def forward(self, image_embeds, text_labels):
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.text_decoder.device)

        input_ids = inputs.input_ids
        print("Input ids before crop:")
        print(input_ids)
        input_ids = input_ids[:, 1:]
        print("Input ids after crop:")
        print(input_ids)

        attention_mask = inputs.attention_mask[:, 1:]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # get the number of tokens until first : (inclusive) and set labels to -100
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == self.tokenizer.convert_tokens_to_ids("###\n"):
                    labels[i, : (j + 1)] = -100
                    break

        input_embeds = self.text_decoder.model.model.embed_tokens(input_ids)
        image_embeds_len = image_embeds.shape[1]
        image_labels = torch.ones(
            (image_embeds.shape[0], image_embeds_len), dtype=torch.long
        ).to(self.text_decoder.device)
        image_labels = image_labels * -100
        input_embeds = torch.cat((image_embeds, input_embeds), dim=1)
        labels = torch.cat((image_labels, labels), dim=1)
        attention_mask = torch.cat(
            (torch.ones_like(image_labels), attention_mask), dim=1
        )
        max_seq_len = 1024
        if input_embeds.shape[1] > max_seq_len:
            input_embeds = input_embeds[:, :max_seq_len, :]
            labels = labels[:, :max_seq_len]
            attention_mask = attention_mask[:, :max_seq_len]

        outputs = self.text_decoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        return loss

    @torch.no_grad()
    def generate(self, image_embeds, text_labels, **kwargs):
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.text_decoder.device)
        input_ids = inputs.input_ids
        input_ids = input_ids[:, 1:]

        input_embeds = self.text_decoder.model.model.embed_tokens(input_ids)

        input_embeds = torch.cat((image_embeds, input_embeds), dim=1)
        outputs = self.text_decoder.generate(inputs_embeds=input_embeds, **kwargs)
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts


class Clip3DForTextGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_image = ModifiedImageEncoder()
        self.decode_text = TextDecoder()
        self.adapter = Adapter(2048, 4096)
        for param in self.encode_image.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        with torch.no_grad():
            image_features = self.encode_image(image)
            image_features = self.adapter(image_features)
            loss = self.decode_text(image_features, text)
            return loss

    @torch.no_grad()
    def generate(self, image, text_labels, **kwargs):
        image_features = self.encode_image(image)
        image_features = self.adapter(image_features)
        texts = self.decode_text.generate(image_features, text_labels, **kwargs)
        return texts


# ==============================================================================
# 6. LOAD / MAIN ENTRY (merlin/models/load.py)
# ==============================================================================

REPO_ID = "stanfordmimi/Merlin"
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "builder": MerlinArchitecture,
        "checkpoint": "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
    },
    "report_generation": {
        "builder": Clip3DForTextGeneration,
        "checkpoint": "resnet_gpt2_best_stanford_report_generation_average.pt",
    },
    "five_year_disease_prediction": {
        "builder": MerlinArchitecture,
        "checkpoint": "resnet_clinical_longformer_five_year_disease_prediction.pt",
    },
}

class Merlin(nn.Module):
    def __init__(
        self,
        ImageEmbedding: bool = False,
        PhenotypeCls: bool = False,
        RadiologyReport: bool = False,
        FiveYearPred: bool = False,
    ):
        super(Merlin, self).__init__()

        # If both are True, raise an error
        if sum([ImageEmbedding, PhenotypeCls, FiveYearPred]) > 1:
            raise ValueError(
                "ImageEmbedding and PhenotypeCls and FiveYearPred cannot be True at the same time."
            )

        self.task = (
            "report_generation"
            if RadiologyReport
            else ("five_year_disease_prediction" if FiveYearPred else "default")
        )

        self._config = MODEL_CONFIGS[self.task]

        # Pass through the flags needed by the underlying model builders
        model_kwargs = (
            {
                "ImageEmbedding": ImageEmbedding,
                "PhenotypeCls": PhenotypeCls,
                "FiveYearPred": FiveYearPred,
            }
            if not RadiologyReport
            else {}
        )
        self.model = self._load_model(**model_kwargs)

    def _load_model(self, **kwargs) -> nn.Module:
        """
        Downloads the correct checkpoint and constructs the appropriate model.
        """
        checkpoint_name = self._config["checkpoint"]
        model_builder = self._config["builder"]

        # Download checkpoint to local directory
        local_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints"
        )
        checkpoint_path = os.path.join(local_dir, checkpoint_name)
        self._download_checkpoint(filename=checkpoint_name, local_dir=local_dir)

        # Build the model
        model = model_builder(**kwargs)

        print(f"Loading checkpoint for '{self.task}' task from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if self.task == "five_year_disease_prediction":
            model.encode_image.i3_resnet.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict)

        return model

    def _download_checkpoint(self, filename: str, local_dir: str):
        if not os.path.exists(os.path.join(local_dir, filename)):
            print(f"Downloading {filename} from Hugging Face Hub...")
            download_file(repo_id=REPO_ID, filename=filename, local_dir=local_dir)

    def forward(self, *args, **kwargs):
        """Delegates the forward call to the underlying model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generates text if the model is in RadiologyReport mode.
        Passes all arguments to the underlying model's generate method.
        """
        if self.task != "report_generation":
            raise AttributeError(
                "The 'generate' method is only available when RadiologyReport=True."
            )
        # Delegate the call to the actual text generation model
        return self.model.generate(*args, **kwargs)