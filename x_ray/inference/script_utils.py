__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse
import hashlib
import logging
import functools
import os
import random
import time
from typing import Optional

SCRIPT_PATH = os.path.basename(__file__)

DEFAULT_LOGGING = "INFO"
DEFAULT_TIME_STR = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
DEFAULT_LOG_DIR = 'logs'

RANDOM_STRING_SEED = 'f39sj)3j09ja0e8f1as98u!98auf-b23bacxmza9820h35m./9'


try:
    random = random.SystemRandom()
    using_sysrandom = True
except NotImplementedError:
    import warnings
    warnings.warn('A secure pseudo-random number generator is not available '
                  'on your system. Falling back to Mersenne Twister.')
    using_sysrandom = False


def parse_args(parser: argparse.ArgumentParser) -> dict:
    args, _ = parser.parse_known_args()
    args = vars(args)
    return args        


def get_random_string(length=12,
                      allowed_chars='abcdefghijklmnopqrstuvwxyz'
                                    'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """
    Returns a securely generated random string.

    The default length of 12 with the a-z, A-Z, 0-9 character set returns
    a 71-bit value. log_2((26+26+10)^12) =~ 71 bits
    """
    if not using_sysrandom:
        # This is ugly, and a hack, but it makes things better than
        # the alternative of predictability. This re-seeds the PRNG
        # using a value that is hard for an attacker to predict, every
        # time a random string is required. This may change the
        # properties of the chosen random sequence slightly, but this
        # is better than absolute predictability.
        random.seed(
            hashlib.sha256(
                ("%s%s%s" % (
                    random.getstate(),
                    time.time(),
                    RANDOM_STRING_SEED)).encode('utf-8')
            ).digest())
    return ''.join(random.choice(allowed_chars) for _ in range(length))


def tuple_to_args(fn, is_method: Optional[bool] = False):
    @functools.wraps(fn)
    def tuple_to_args_method_wrapper(self, tup: tuple, *args, **kwargs):
        if isinstance(tup, tuple):
            return fn(self, *tup, *args, **kwargs)
        return fn(self, tup, *args, **kwargs)


    @functools.wraps(fn)
    def tuple_to_args_wrapper(tup: tuple, *args, **kwargs):
        if isinstance(tup, tuple):
            return fn(*tup, *args, **kwargs)
        return fn(tup, *args, **kwargs)        


    if is_method:
        return tuple_to_args_method_wrapper
    return tuple_to_args_wrapper


def async_tuple_to_args(fn, is_method: Optional[bool] = False):
    @functools.wraps(fn)
    async def tuple_to_args_method_wrapper(self, tup: tuple, *args, **kwargs):
        if isinstance(tup, tuple):
            return await fn(self, *tup, *args, **kwargs)
        return await fn(self, tup, *args, **kwargs)


    @functools.wraps(fn)
    async def tuple_to_args_wrapper(tup: tuple, *args, **kwargs):
        if isinstance(tup, tuple):
            return await fn(*tup, *args, **kwargs)
        return await fn(tup, *args, **kwargs)
    

    if is_method:
        return tuple_to_args_method_wrapper
    return tuple_to_args_wrapper    


def arg_is_true(arg_str: str) -> bool:
    return arg_str in (True, "True", "TRUE", "true", "T", "t")


def arg_is_false(arg_str: str) -> bool:
    return arg_str in (False, "False", "FALSE", "false", "F", "f")


def get_args(
    script_path, logging_level = DEFAULT_LOGGING, time_str = DEFAULT_TIME_STR, 
    log_dir=DEFAULT_LOG_DIR, log_filepath = None, verbose=True, 
    secret_keys = list(), format_str = "%(asctime)s | %(message)s", **kwargs
):

    if not log_filepath:
        log_filepath = os.path.join(
            log_dir, f"{script_path}_{time_str}.log"
        ).replace('\\', '/')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging",
        default=logging_level,
        type=str
    )
    parser.add_argument(
        "--log-filepath",
        default=log_filepath,
        help="Path to a local file to which log messages will be written.",
        type=str
    )
    p_args, _ = parser.parse_known_args()
    p_args = vars(p_args)

    # HERE you can append any keys which you don't want to write to a log file
    secret_keys.append("credentials")
    secret_keys.append("google_application_credentials")
    secret_keys.append("gcs_credentials")
    secret_keys.append("planet_api_key")
    secret_keys.append("PLANET_API_KEY")

    p_args = {**p_args, **kwargs}
    p_args['script_path'] = script_path

    if p_args['logging'] in ("INFO", "info", "Info", "I", "i"):
        logging.getLogger().setLevel(logging.INFO)
        log_filepath = p_args['log_filepath']
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

        
        logFormatter = logging.Formatter(format_str)
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(log_filepath)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        # consoleHandler = logging.StreamHandler()
        # consoleHandler.setFormatter(logFormatter)
        # rootLogger.addHandler(consoleHandler)

        if verbose:
            for key, value in p_args.items():
                if key in secret_keys:
                    logging.info(f"{key}: <REDACTED>")
                else:
                    logging.info(f"{key}: {value}")
    return p_args


if __name__ == "__main__":
    d = get_args(script_path=SCRIPT_PATH)
    logging.info(f'Parameters received: \n {d}')
