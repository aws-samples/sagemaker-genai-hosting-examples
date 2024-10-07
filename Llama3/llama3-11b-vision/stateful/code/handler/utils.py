import time
import io
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

@measure_time
def download_image(url):
    """
    Downloads an image from the given URL and converts it to an RGB image.
    
    Parameters:
    url (str): The URL of the image to download.
    
    Returns:
    PIL.Image: The downloaded and converted image.
    """
    # Download the image from the URL
    response = requests.get(url)
    
    # Load the image using PIL
    image = Image.open(io.BytesIO(response.content))
    
    # Convert the image to RGB mode
    rgb_image = image.convert('RGB')
    
    return rgb_image


if __name__ == "__main__":
    image_url="https://images.pexels.com/photos/1519753/pexels-photo-1519753.jpeg"
    download_image(image_url)