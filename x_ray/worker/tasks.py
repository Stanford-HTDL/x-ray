__author__ = "Richard Correro (richard@richardcorrero.com)"

import hashlib
import io
import os
import zipfile
from datetime import datetime, timedelta
from typing import Optional

import torch
from google.cloud import storage

from ..celery_config.celery import celery_app
from ..inference.load_model_from_file import load_model
from ..inference.predict import predict
from ..utils import get_datetime

# IDS_PATH: str = os.environ["IDS_PATH"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
MODEL_PATH: str = os.environ["MODEL_PATH"]
MODEL_UID: str = os.environ["MODEL_UID"]
DEVICE: str = os.environ["DEVICE"]

GCS_CREDS_PATH: str = os.environ["GCS_CREDS_PATH"]
BUCKET_NAME: str = os.environ["GCS_BUCKET_NAME"]
OUTPUT_FILE_TYPE: str = os.environ["OUTPUT_FILE_TYPE"]
# OUTPUT_FILE_TYPE: str = "application/zip" # Hard code (for now)

SIGNED_URL_EXPIRATION_MINUTES: int = int(os.environ["SIGNED_URL_EXPIRATION_MINUTES"])

# Load model into memory here so that its not reloaded every time task is called
MODEL: torch.nn.Module = load_model(
    model_name=MODEL_NAME, model_filepath=MODEL_PATH, device=DEVICE
)

CLIENT: storage.Client = storage.Client.from_service_account_json(GCS_CREDS_PATH)


# def get_files_as_bytes(filenames: Iterable[str]) -> io.BytesIO:
#     # Create a BytesIO instance with the .zip file
#     zip_file_data = io.BytesIO()
#     with zipfile.ZipFile(zip_file_data, 'w') as zipf:
#         for filename in filenames:
#             zipf.write(filename)
#     zip_file_data.seek(0)
#     return zip_file_data


def hash_string(string: str) -> str:
    """Hash a string using the SHA-256 algorithm."""
    # Encode the string to bytes
    string_bytes = string.encode("utf-8")

    # Create a hash object using the SHA-256 algorithm
    hash_object = hashlib.sha256()

    # Update the hash object with the bytes of the string
    hash_object.update(string_bytes)

    # Get the hashed value as a hexadecimal string
    hashed_string = hash_object.hexdigest()

    return hashed_string


def zip_directory_as_bytes(directory_path: str) -> io.BytesIO:
    """Create a zip archive of a directory."""
    zip_file_data: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(zip_file_data, "w") as zip_file:
        # Iterate over all the files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                # Create the full file path by joining the directory and file paths
                file_path: str = os.path.join(root, file)
                # Add the file to the zip archive, using the relative path inside the directory
                zip_file.write(file_path, os.path.relpath(file_path, directory_path))
    zip_file_data.seek(0)
    return zip_file_data


def upload_file_to_gcs(
    zip_file_data: io.BytesIO, blob_name: str, client: storage.Client, 
    bucket_name: str, content_type: Optional[str] = 'application/zip',
) -> None:
    # Upload the .zip file to GCS with authentication
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(zip_file_data, content_type=content_type)


def get_signed_url(
    blob_name: str, client: storage.Client, bucket_name: str,
    exp_minutes: Optional[int] = 60
) -> str:
    # Generate a signed URL with expiration time
    expiration_time = datetime.utcnow() + timedelta(minutes=exp_minutes)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    signed_url = blob.generate_signed_url(
        expiration=expiration_time, method='GET'
    )
    return signed_url


def blob_exists(bucket_name: str, blob_name: str, gcs_creds_path: str):
    """Check whether a blob exists in a GCS bucket."""
    # Instantiate a client
    storage_client = storage.Client.from_service_account_json(gcs_creds_path)

    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Check if the blob exists
    return blob.exists()


@celery_app.task(name="analyze")
def analyze(
    start: str, stop: str, target_geojson: str, process_uid: str, **kwargs
) -> dict:
    start_datetime: datetime = get_datetime(start)
    start_formatted: str = start_datetime.strftime('%Y_%m')
    stop_datetime: datetime = get_datetime(stop)
    stop_formatted: str = stop_datetime.strftime('%Y_%m')

    # if id not in VALID_IDS:
    #     raise IDNotFoundError(
    #         f"ID {id} not in valid IDs. Valid IDs may be found at {IDS_PATH}."
    #     )

    geojson_hash: str = hash_string(target_geojson)

    save_dir_path: str = \
        f"{MODEL_UID}/results/{geojson_hash}/{start_formatted}/{stop_formatted}"
    
    results_blob_name: str = f"{save_dir_path}/results.zip"

    # Check whether results have already been generated – if so, do not rerun analysis
    if not blob_exists(
        bucket_name=BUCKET_NAME, blob_name=results_blob_name, 
        gcs_creds_path=GCS_CREDS_PATH
    ):
        predict(model=MODEL, device=DEVICE, 
            uid=process_uid, target_geojson=target_geojson, 
            start=start_formatted, stop_datetime=stop_formatted, 
            save_dir_path=save_dir_path
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

    # blob_name = f"results/id_{id}"

    # blob_name: str = "dir_0/dir_1/dir_2/lorem_ipsum.zip"
    # content_type: str = "application/zip"
    # exp_minutes: int = 60
    # signed_url: str = get_signed_url(
    #     filenames=filenames, gcs_creds_path=GCS_CREDS_PATH, 
    #     bucket_name=BUCKET_NAME, blob_name=blob_name, 
    #     content_type=OUTPUT_FILE_TYPE, exp_minutes=SIGNED_URL_EXPIRATION_MINUTES
    # )

    return {"url": signed_url}
