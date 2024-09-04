import functools
import time
from typing import Callable

from data_types import InferenceSession
from torch import Tensor

import boto3
from botocore.exceptions import ClientError
import io
from urllib.parse import urlparse
from PIL import Image
import requests
import logging

logger = logging.getLogger(__name__)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"MYLOGS-UTILS-TIME: Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

def download_image(url):
    print(f"MYLOGS-UTILS: image url is {url}")
    return Image.open(requests.get(url, stream=True).raw)
    
def get_session_state_size(session_state: InferenceSession) -> int:
    """
    Compute the session state size in bytes.

    https://discuss.pytorch.org/t/how-to-know-the-memory-allocated-for-a-tensor-on-gpu/28537/2
    """
    size = 0
    for _, v in session_state.state.items():
        for _, t in v.items():
            if isinstance(t, dict):
                for _, it in t.items():
                    size += __get_tensor_size(it)
            else:
                size += __get_tensor_size(t)
    return size


def __get_tensor_size(t: Tensor) -> int:
    """
    Compute the tensor size in bytes.

    https://discuss.pytorch.org/t/how-to-know-the-memory-allocated-for-a-tensor-on-gpu/28537/2
    """
    return t.element_size() * t.nelement()
