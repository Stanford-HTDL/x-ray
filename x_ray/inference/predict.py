__author__ = "Richard Correro (richard@richardcorrero.com)"

import logging
import os
import random
import time
from typing import Dict

import torch
# from models import FasterRCNN
from .pred_processors import (ConvLSTMCProcessor, ObjectDetectorProcessor,
                             Processor, ResNetProcessor)
from .script_utils import get_args, get_random_string

from ..utils import get_env_var_with_default

SCRIPT_PATH = os.path.basename(__file__)

# DEFAULT_MODEL_NAME = FasterRCNN.__name__
DEFAULT_PRED_PROCESSOR = ObjectDetectorProcessor.__name__
DEFAULT_EXPERIMENT_DIR = "experiments/"
DEFAULT_DEVICE = "CUDA if available else CPU"
DEFAULT_CHANNEL_AXIS = 1
DEFAULT_EXPERIMENT_DIR: str = "experiments/"
DEFAULT_USE_TIME_STR_EXPERIMENT_DIR: str = True

DEFAULT_SEED = 8675309 # (___)-867-5309

# # REMEMBER to update as new models are added!
# MODELS = {
#     FasterRCNN.__name__: FasterRCNN
# }

PRED_PROCESSORS = {
    ConvLSTMCProcessor.__name__: ConvLSTMCProcessor,
    ResNetProcessor.__name__: ResNetProcessor,
    ObjectDetectorProcessor.__name__: ObjectDetectorProcessor
}


def parse_args():
    SEED: int = int(get_env_var_with_default(
        "SEED",
        default=DEFAULT_SEED,
    ))
    # MODEL: str = get_env_var_with_default(
    #     "MODEL",
    #     # default=DEFAULT_MODEL_NAME
    # )
    PRED_PROCESSOR: str = get_env_var_with_default(
        "PRED_PROCESSOR",
        default=DEFAULT_PRED_PROCESSOR
    )    
    MODEL_FILEPATH: str = os.environ["MODEL_FILEPATH"]
    DEVICE: str = get_env_var_with_default(
        "DEVICE",
        default=DEFAULT_DEVICE
    )
    CHANNEL_AXIS: int = int(get_env_var_with_default(
        "CHANNEL_AXIS",
        default=DEFAULT_CHANNEL_AXIS,
    ))
    EXPERIMENT_DIR: str = get_env_var_with_default(
        "EXPERIMENT_DIR",
        default=DEFAULT_EXPERIMENT_DIR
    )
    DATA_DIR: str = os.environ["DATA_DIR"] 
    ID: str = os.environ["ID"]
    USE_TIME_STR_EXPERIMENT_DIR: bool = bool(get_env_var_with_default(
        "USE_TIME_STR_EXPERIMENT_DIR",
        default=DEFAULT_USE_TIME_STR_EXPERIMENT_DIR
    ))
    args: dict = {
        "seed": SEED,
        "model": MODEL,
        "pred_processor": PRED_PROCESSOR,
        "model_filepath": MODEL_FILEPATH,
        "device": DEVICE,
        "channel_axis": CHANNEL_AXIS,
        "experiment_dir": EXPERIMENT_DIR,
        "data_dir": DATA_DIR,
        "id": ID,
        "use_time_str_experiment_dir": USE_TIME_STR_EXPERIMENT_DIR
    }
    return args


def predict(
    model, device: str, uid: str, target_geojson_dict: Dict,
    start: str, stop: str, save_dir_path: str
) -> None:
    args = vars(parse_args())

    seed = args["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)        

    if not args["id"]:
        experiment_id = get_random_string()
    else:
        experiment_id = args["id"]
    experiment_super_dir = args["experiment_dir"]
    use_time_str_in_experiment_dir: bool = args["use_time_str_experiment_dir"]
    time_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())   
    if use_time_str_in_experiment_dir: 
        experiment_dir = os.path.join(
            experiment_super_dir, experiment_id, f"{time_str}/"
        ).replace("\\", "/")
    else:
        experiment_dir = os.path.join(
            experiment_super_dir, experiment_id + "/",
        ).replace("\\", "/")                
    log_dir = os.path.join(experiment_dir, 'logs/').replace("\\", "/")
    # save_dir = os.path.join(
    #     experiment_dir, "model_checkpoints/"
    # ).replace("\\", "/")
    os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(save_dir, exist_ok=True)
    log_filepath = os.path.join(
        log_dir, f"{SCRIPT_PATH}_{time_str}_{experiment_id}.log"
    ).replace('\\', '/')

    device = args["device"]
    if device == DEFAULT_DEVICE:
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ("CUDA", "Cuda", "cuda"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # logging.info(f'Using device {device}')       

    # model_name = args["model"]
    # model: torch.nn.Module = MODELS[model_name]()    

    model.to(device=device)

    model_filepath: str = args["model_filepath"]
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model_num_channels = model.args["num_channels"] # A constraint on the Model class
    channel_axis: int = args["channel_axis"]    

    # logging.info(f'Model loaded from {model_filepath}')

    pred_processor_name: str = args["pred_processor"]
    pred_processor: Processor = PRED_PROCESSORS[pred_processor_name](save_dir=experiment_dir)

    # Note: You CANNOT place a `logging.info(...)` command before calling `get_args(...)`
    args = get_args(
        script_path=SCRIPT_PATH, log_filepath=log_filepath, 
        **args, **model.args, **pred_processor.args, 
        experiment_id=experiment_id, time=time_str
    )

    logging.info(f'Using device {device}')
    logging.info(f'Model loaded from {model_filepath}')     

    data_dir = args["data_dir"]

    # samples = sample_generator(dir_path=data_dir)
    samples = pred_processor.make_samples(dir_path=data_dir)

    model.eval()
    sample_idx: int = 0
    for sample in samples:
        sample_idx += 1
        X: torch.Tensor = sample["X"]
        X: torch.Tensor = X.to(device=device, dtype=torch.float32)
        X_num_channels = X.shape[channel_axis]
        assert X_num_channels == model_num_channels, \
            f"Network has been defined with {model_num_channels}" \
            f"input channels, but loaded images have {X_num_channels}" \
            "channels. Please check that the images are loaded correctly."        
        logging.info(f"Generating predictions for sample {sample_idx}...")
        with torch.no_grad():
            pred = model(X)
        logging.info(f"Saving predictions for sample {sample_idx}...")
        pred_processor.save_results(input=sample, output=pred)
        # save_preds(input=sample, output=pred)    
    logging.info(
        """
                ================
                =              =
                =     Done.    =
                =              =
                ================
        """
    )


def main():
    predict()


if __name__ == "__main__":
    main()
