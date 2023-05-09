FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update --fix-missing && DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
   build-essential \
   python3 \
   python3-dev \
   python3-pip

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./_gcs_creds.json /_gcs_creds.json

COPY ./models /models

COPY ./x_ray /x_ray

CMD celery -A x_ray.celery_config.celery worker -l INFO
