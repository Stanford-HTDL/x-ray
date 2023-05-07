__author__ = "Richard Correro (richard@richardcorrero.com)"

import logging
import os
import time
from typing import List

import torch

from .pred_processors import ObjectDetectorProcessor, Processor
from .script_utils import get_args

SCRIPT_PATH = "predict"

PRED_PROCESSORS = {
    ObjectDetectorProcessor.__name__: ObjectDetectorProcessor
}


def prep_for_prediction(
    model: torch.nn.Module, id: str, save_dir_path: str, bbox_threshold: float, 
    device: torch.device, pred_processor_name: str, 
) -> Processor:
    time_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())    
    os.makedirs(save_dir_path, exist_ok=True)             
    log_dir = os.path.join(save_dir_path, 'logs/').replace("\\", "/")
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(
        log_dir, f"{id}.log"
    ).replace('\\', '/')

    # pred_processor_name: str = args["pred_processor"]
    pred_processor: Processor = PRED_PROCESSORS[pred_processor_name](
        save_dir=save_dir_path, bbox_threshold=bbox_threshold
    )

    # Note: You CANNOT place a `logging.info(...)` command before calling `get_args(...)`
    get_args(
        script_path=SCRIPT_PATH, log_filepath=log_filepath, 
        **model.args, **pred_processor.args, 
        id=id, time=time_str
    )

    logging.info(f'Using device {device}')

    return pred_processor


def predict(
    model: torch.nn.Module, device: torch.device, target_geojson_strs: List[str],
    start: str, stop: str,
    pred_processor: Processor
) -> None:
    samples = pred_processor.make_samples(target_geojson_strs=target_geojson_strs, start=start, end=stop)
    # model_num_channels: int = model.args["num_channels"]
    model.eval()
    sample_idx: int = 0
    for sample in samples:
        sample_idx += 1
        X: torch.Tensor = sample["X"]
        X: torch.Tensor = X.to(device=device, dtype=torch.float32)
        # X_num_channels = X.shape[channel_axis]
        # assert X_num_channels == model_num_channels, \
        #     f"Network has been defined with {model_num_channels}" \
        #     f"input channels, but loaded images have {X_num_channels}" \
        #     "channels. Please check that the images are loaded correctly."        
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
