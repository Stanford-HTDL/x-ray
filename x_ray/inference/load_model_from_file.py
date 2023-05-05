__author__ = "Richard Correro (richard@richardcorrero.com)"

import torch

from .models import FasterRCNN

MODELS = {
    FasterRCNN.__name__: FasterRCNN
}

def get_device(device: str) -> torch.device:
    if device == "CUDA if available else CPU":
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ("CUDA", "Cuda", "cuda"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_model(model_name: str, model_filepath: str, device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = MODELS[model_name]()
    model.to(device=device)
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model
