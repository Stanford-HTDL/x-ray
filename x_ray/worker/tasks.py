__author__ = "Richard Correro (richard@richardcorrero.com)"

import io
import os
from datetime import datetime

import torch
from google.cloud import storage

from ..celery_config.celery import celery_app
from ..inference.load_model_from_file import get_device, load_model
from ..inference.pred_processors import Processor
from ..inference.predict import predict, prep_for_prediction
from ..utils import (blob_exists, get_datetime, get_signed_url, hash_string,
                     upload_file_to_gcs, zip_directory_as_bytes)

torch.set_num_threads(1)

# IDS_PATH: str = os.environ["IDS_PATH"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
MODEL_PATH: str = os.environ["MODEL_PATH"]
MODEL_UID: str = os.environ["MODEL_UID"]

PRED_PROCESSOR: str = os.environ["PRED_PROCESSOR"]
DEVICE: str = os.environ["DEVICE"]

GCS_CREDS_PATH: str = os.environ["GCS_CREDS_PATH"]
BUCKET_NAME: str = os.environ["GCS_BUCKET_NAME"]
OUTPUT_FILE_TYPE: str = os.environ["OUTPUT_FILE_TYPE"]
# OUTPUT_FILE_TYPE: str = "application/zip" # Hard code (for now)

SIGNED_URL_EXPIRATION_MINUTES: int = int(os.environ["SIGNED_URL_EXPIRATION_MINUTES"])

DEVICE: torch.device = get_device(device=DEVICE)
# Load model into memory here so that its not reloaded every time task is called
MODEL: torch.nn.Module = load_model(
    model_name=MODEL_NAME, model_filepath=MODEL_PATH, device=DEVICE
)

CLIENT: storage.Client = storage.Client.from_service_account_json(GCS_CREDS_PATH)


@celery_app.task(name="analyze")
def analyze(
    start: str, stop: str, target_geojson: str, process_uid: str
) -> dict:
    start_datetime: datetime = get_datetime(start)
    start_formatted: str = start_datetime.strftime('%Y_%m')

    stop_datetime: datetime = get_datetime(stop)
    stop_formatted: str = stop_datetime.strftime('%Y_%m')

    geojson_hash: str = hash_string(target_geojson)

    save_dir_path: str = \
        f"results/{MODEL_UID}/{geojson_hash}/{start_formatted}/{stop_formatted}"
    
    results_blob_name: str = f"{save_dir_path}/results.zip"

    # Check whether results have already been generated;
    # if so, do not rerun analysis
    if not blob_exists(
        bucket_name=BUCKET_NAME, blob_name=results_blob_name, 
        gcs_creds_path=GCS_CREDS_PATH
    ):
        pred_processor: Processor = prep_for_prediction(
            model=MODEL, id=process_uid, save_dir_path=save_dir_path, 
            device=DEVICE, pred_processor_name=PRED_PROCESSOR
        )
        predict(model=MODEL, device=DEVICE, target_geojson_strs=[target_geojson], 
            pred_processor=pred_processor, start=start_formatted, stop=stop_formatted
        )
        zip_file_data: io.BytesIO = zip_directory_as_bytes(directory_path=save_dir_path)
        upload_file_to_gcs(
            zip_file_data=zip_file_data, blob_name=results_blob_name, 
            client=CLIENT, bucket_name=BUCKET_NAME, content_type=OUTPUT_FILE_TYPE
        )
    signed_url: str = get_signed_url(
        blob_name=results_blob_name, client=CLIENT, bucket_name=BUCKET_NAME,
        exp_minutes=SIGNED_URL_EXPIRATION_MINUTES
    )

    return {"url": signed_url}
