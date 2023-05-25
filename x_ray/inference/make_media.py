__author__ = "Richard Correro (richard@richardcorrero.com)"

import logging
import os
import time
from typing import List

from .pred_processors import PlanetMediaMaker, Processor
from .script_utils import get_args

SCRIPT_PATH = "make_media"

PRED_PROCESSORS = {
    PlanetMediaMaker.__name__: PlanetMediaMaker
}


def prep_for_media(
    id: str, save_dir_path: str, pred_processor_name: str, 
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
        save_dir=save_dir_path, 
    )

    # Note: You CANNOT place a `logging.info(...)` command before calling `get_args(...)`
    get_args(
        script_path=SCRIPT_PATH, log_filepath=log_filepath, 
        **pred_processor.args, id=id, time=time_str
    )
    return pred_processor

def make_media_samples(
    target_geojson_strs: List[str],start: str, stop: str, pred_processor: Processor
) -> None:
    samples = pred_processor.make_samples(target_geojson_strs=target_geojson_strs, start=start, end=stop)
    for _ in samples:
        pass
    logging.info(
        """
                ================
                =              =
                =     Done.    =
                =              =
                ================
        """
    )
