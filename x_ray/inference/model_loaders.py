__author__ = "Richard Correro (richard@richardcorrero.com)"


from typing import Callable

from torch import nn
from torchvision import models
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models import detection
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor, _validate_trainable_layers)
from torchvision.models.detection.faster_rcnn import model_urls
from torchvision.ops import misc as misc_nn_ops

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}


# Adapted From: https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn
def fasterrcnn_resnet_fpn(
    backbone_name: str, pretrained=False, progress=True, num_classes=2, 
    pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
    """
    Constructs a Faster R-CNN model with a ResNet-*-FPN backbone. Unlike the original
    code, this function allows the user to specify _which_ resnet model to use as
    backbone.

    Args:
        backbone_name (str): Name of (residual network) module used as backbone
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    resnet: Callable = BACKBONES[backbone_name]

    backbone: nn.Module = resnet(
        pretrained=pretrained_backbone, progress=progress, 
        norm_layer=misc_nn_ops.FrozenBatchNorm2d
    )
    # backbone = resnet_fpn_backbone(backbone_name=backbone_name, trainable_backbone_layers=trainable_backbone_layers)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = detection.FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model
