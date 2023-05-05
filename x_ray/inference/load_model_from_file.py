__author__ = "Richard Correro (richard@richardcorrero.com)"

import torch

from .models import FasterRCNN

MODELS = {
    FasterRCNN.__name__: FasterRCNN
}


def load_model(model_name: str, model_filepath: str, device: str) -> torch.nn.Module:
    model: torch.nn.Module = MODELS[model_name]()    
    model.to(device=device)
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model
