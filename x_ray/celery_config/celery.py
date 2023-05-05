__author__ = "Richard Correro (richard@richardcorrero.com)"

from celery import Celery

import os

CELERY_APP_NAME: str = os.environ["CELERY_APP_NAME"]
CELERY_APP_INCLUDE: str = os.environ["CELERY_APP_INCLUDE"]
APP_BROKER_URI: str = os.environ["APP_BROKER_URI"]
APP_BACKEND_URI: str = os.environ["APP_BACKEND_URI"]

celery_app = Celery(
    CELERY_APP_NAME, broker=APP_BROKER_URI, backend=APP_BACKEND_URI, 
    include=[CELERY_APP_INCLUDE]
)

# Optional configuration, see the application user guide.
celery_app.conf.update(
    result_expires=3600,
)


if __name__ == '__main__':
    celery_app.start()
