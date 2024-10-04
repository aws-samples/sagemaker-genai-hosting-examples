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
import os, sys
from PIL import Image as PIL_Image

logger = logging.getLogger(__name__)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"MYLOGS-UTILS-TIME: Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

def download_image(image_path):
    """
    Open and convert an image from the specified path.
    """
    print(f"MYLOGS-UTILS: image url is {image_path}")
    if not os.path.exists(image_path):
        print(f"The image file '{image_path}' does not exist.")
        sys.exit(1)
    with open(image_path, "rb") as f:
        return PIL_Image.open(f).convert("RGB") 
