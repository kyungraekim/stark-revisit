import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

from export.portable.error import ExportError
from lib.models.stark.position_encoding import build_position_encoding
from export.portable import resnet as resnet_module
from utils.misc import is_main_process


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool,
                 net_type="resnet"):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name:
                parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
                # print('freeze %s'%name)
        if return_interm_layers:
            if net_type == "resnet":
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2"}  # stride = 4, 8, 16
            else:
                raise ValueError()
        else:
            if net_type == "resnet":
                return_layers = {'layer3': "0"}  # stride = 16
            else:
                raise ValueError()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)  # method in torchvision
        self.num_channels = num_channels

    def forward(self, x, m):
        return [*self.body(x)]


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    # Resnet101, true, false, false, true, None
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 freeze_bn: bool,
                 ckpt_path=None):
        if "resnet" in name:
            norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
            # here is different from the original DETR because we use feature from block3
            backbone = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, dilation, False],
                pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
            num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
            net_type = "resnet"
        else:
            raise ValueError()
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, net_type=net_type)


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(self, x, m):
        xs = self.backbone(x, m)
        pos = []
        for x in xs:
            pos.append(self.position_embedding(x).to(x.tensors.dtype))
        return xs, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    return_interm_layers = cfg.MODEL.PREDICT_MASK
    if cfg.MODEL.BACKBONE.TYPE == "RepVGG-A0":
        raise ExportError("RepVGG-A0")
    else:
        ckpt_path = None
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                        cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN, ckpt_path)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
