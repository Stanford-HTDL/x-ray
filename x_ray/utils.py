__author__ = "Richard Correro (richard@richardcorrero.com)"

from datetime import datetime
from typing import Any
import os

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
        value = default
    return value
