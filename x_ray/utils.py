__author__ = "Richard Correro (richard@richardcorrero.com)"

import hashlib
import io
import os
import zipfile
from datetime import datetime, timedelta
from typing import Any, Optional

from google.cloud import storage

from .exceptions import MalformedDateStringError


def get_datetime(time_str: str) -> datetime:
    try: 
        dt: datetime = datetime.strptime(time_str[:7], '%Y_%m')
    except:
        raise MalformedDateStringError(
            f"Time string must be in %Y_%m format. Received string {time_str}."
        )
    return dt


def get_env_var_with_default(env_var_name: str, default: Any) -> Any:
    try:
        value: str = os.environ[env_var_name]
    except KeyError:
        print(f"Environment variable {env_var_name} not found. Using default value {default}.")
        value = default
    return value


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
                # Create the full file path by joining the directory and 
                # file paths
                file_path: str = os.path.join(root, file)
                # Add the file to the zip archive, using the relative path 
                # inside the directory
                zip_file.write(file_path, os.path.relpath(
                    file_path, directory_path
                    )    
                )
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
