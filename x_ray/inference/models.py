__author__ = "Richard Correro (richard@richardcorrero.com)"

import torch.nn as nn

from ..utils import get_env_var_with_default
from .model_loaders import fasterrcnn_resnet_fpn
from .script_utils import arg_is_true


class FasterRCNN(nn.Module):
    __name__ = "FasterRCNN"

    IS_OBJECT_DETECTOR = True
    # MODEL_NAME = "fasterrcnn_resnet_fpn"

    DEFAULT_BACKBONE_NAME: str = "resnet50"
    DEFAULT_NUM_CHANNLES = 3
    DEFAULT_NUM_CLASSES = 2
    DEFAULT_TRAINABLE_LAYERS = 0
    DEFAULT_PRETRAINED = False
    DEFAULT_PROGRESS = False 
    DEFAULT_PRETRAINED_BACKBONE = True 
    DEFAULT_MIN_SIZE: int = 224
    DEFAULT_MAX_SIZE: int = 224


    def __init__(self, **kwargs):
        super().__init__()
        args = self.parse_args()
        backbone_name: str = args["backbone"]
        num_channels = int(args["num_channels"])
        assert num_channels == 3, f"Must have `num_channels == 3` for model {self.__name__}."        
        
        num_classes = int(args["num_classes"])
        trainable_layers = int(args["trainable_layers"])
        pretrained: bool = args["pretrained"]
        progress: bool = args["progress"]
        pretrained_backbone: bool = args["pretrained_backbone"]
        min_size: int = int(args["min_size"])
        max_size: int = int(args["max_size"])
        self.args = {**args, **kwargs}

        model = fasterrcnn_resnet_fpn(
            backbone_name=backbone_name,
            pretrained=pretrained, progress=progress, num_classes=num_classes, 
            pretrained_backbone=pretrained_backbone, min_size=min_size, 
            max_size=max_size, trainable_backbone_layers=trainable_layers, 
            **kwargs
        )
        self.model = model


    def parse_args(self):
        BACKBONE_NAME: str = get_env_var_with_default(
            "BACKBONE_NAME", self.DEFAULT_BACKBONE_NAME
        )
        # BACKBONE: str = os.environ["BACKBONE"]

        NUM_CHANNELS: int = int(
            get_env_var_with_default(
                "NUM_CHANNELS", self.DEFAULT_NUM_CHANNLES
            )
        )
        # NUM_CHANNELS: int = int(os.environ["NUM_CHANNELS"])
       
        NUM_CLASSES: int = int(
            get_env_var_with_default(
                "NUM_CLASSES", self.DEFAULT_NUM_CLASSES
            )
        )
        # NUM_CLASSES: int = int(os.environ["NUM_CLASSES"])

        TRAINABLE_LAYERS: int = int(
            get_env_var_with_default(
                "TRAINABLE_LAYERS", self.DEFAULT_TRAINABLE_LAYERS
            )
        )
        # TRAINABLE_LAYERS: int = int(os.environ["RAINABLE_LAYERS"])
        
        PRETRAINED: bool = arg_is_true(
            get_env_var_with_default(
                "PRETRAINED", default=self.DEFAULT_PRETRAINED
            )
        )
        # PRETRAINED: bool = arg_is_true(os.environ["PRETRAINED"])

        PROGRESS: bool = arg_is_true(
            get_env_var_with_default(
                "PROGRESS", self.DEFAULT_PROGRESS
            )
        )
        # PROGRESS: bool = arg_is_true(os.environ["PROGRESS"])

        PRETRAINED_BACKBONE: bool = arg_is_true(
            get_env_var_with_default(
                "PRETRAINED_BACKBONE", self.DEFAULT_PRETRAINED_BACKBONE
            )
        )
        # PRETRAINED_BACKBONE: bool = arg_is_true(os.environ["PRETRAINED_BACKBONE"])
        
        MIN_SIZE: int = int(
            get_env_var_with_default(
                "MIN_SIZE", self.DEFAULT_MIN_SIZE
            )
        )
        # MIN_SIZE: int = int(os.environ["MIN_SIZE"])

        MAX_SIZE: int = int(
            get_env_var_with_default(
                "MAX_SIZE", self.DEFAULT_MAX_SIZE
            )
        )
        # MAX_SIZE: int  = int(os.environ["MAX_SIZE"])
        
        args: dict = {
            "backbone": BACKBONE_NAME,
            "num_channels": NUM_CHANNELS,
            "num_classes": NUM_CLASSES,
            "trainable_layers": TRAINABLE_LAYERS,
            "pretrained": PRETRAINED,
            "progress": PROGRESS,
            "pretrained_backbone": PRETRAINED_BACKBONE,
            "min_size": MIN_SIZE,
            "max_size": MAX_SIZE
        }
        return args

    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
