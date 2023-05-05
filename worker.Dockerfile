FROM python:3.10

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./_gcs_creds.json /_gcs_creds.json

COPY ./x_ray /x_ray

# COPY ./castle/celery_config /celery_config

CMD celery -A x_ray.celery_config.celery worker -l INFO
