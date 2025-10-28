# Standard library imports
import os
import sys
import glob
import random
import base64
import logging
import difflib
import tarfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any
from urllib.parse import urlparse

# Third-party imports
import boto3
import pandas as pd
from PIL import Image  # Changed from PIL import
from tqdm.auto import tqdm
from IPython.display import HTML
from botocore.exceptions import ClientError

def find_latest_version_directory(directory_path: Union[str, Path]) -> str:
    """
    Finds the latest version directory using pathlib for cross-platform safety.
    
    Version directory format: vX-YYYYMMDD-HHMMSS
    Where:
    - X = version number (integer)
    - YYYYMMDD = date of creation
    - HHMMSS = time of creation
    """
    path = Path(directory_path) if isinstance(directory_path, str) else directory_path
    latest_dir = None
    latest_version = -1
    latest_timestamp = None

    for dir_entry in path.iterdir():
        if dir_entry.is_dir() and dir_entry.name.startswith('v'):
            try:
                # Split directory name into components
                version_part, date_str, time_str = dir_entry.name.split('-', 2)
                
                # Extract version number
                version_number = int(version_part[1:])  # Remove 'v' prefix
                
                # Parse datetime
                timestamp = datetime.strptime(
                    f"{date_str} {time_str}", 
                    "%Y%m%d %H%M%S"
                )

                # Update latest version
                if (version_number > latest_version or 
                    (version_number == latest_version and 
                     timestamp > latest_timestamp)):
                    latest_version = version_number
                    latest_timestamp = timestamp
                    latest_dir = dir_entry.name
                    
            except (ValueError, IndexError):
                continue  # Skip invalid format

    if not latest_dir:
        raise FileNotFoundError(f"No valid version directories found in {path}")
        
    return str(latest_dir)





# Alternative version that reads the entire file if memory allows
def find_best_model_checkpoint(file_path):
    # Read the JSONL file
    df = pd.read_json(file_path, lines=True)
    
    # Find the last non-null best_model_checkpoint
    if 'best_model_checkpoint' in df:
        valid_checkpoints = df[df['best_model_checkpoint'].notna()]
        if not valid_checkpoints.empty:
            return valid_checkpoints.iloc[-1]['best_model_checkpoint']
    
    return None

def get_latest_sagemaker_training_job(training_job_name_prefix, sagemaker_client=None):
    if not sagemaker_client:
        sagemaker_client = boto3.client('sagemaker')

    # nameContains must have length less than or equal to 63
    training_job_name_prefix = training_job_name_prefix[:63]
    
    response = sagemaker_client.list_training_jobs(
        NameContains=training_job_name_prefix,
        SortBy='CreationTime',
        SortOrder='Descending',
        StatusEquals='Completed',
        MaxResults=100,
    )
    
    if not response['TrainingJobSummaries']:
        raise ValueError("No latest fine-tuning found. Did your fine-tuning finish?")
        
    # Get the most recent job name
    job_name = response['TrainingJobSummaries'][0]['TrainingJobName']
    
    # Get the training job details
    job_description = sagemaker_client.describe_training_job(
        TrainingJobName=job_name
    )
    return job_description


def get_s3_suffix(s3_uri: str) -> str:
    parsed = urlparse(s3_uri)
    return parsed.path.lstrip('/')





def merge_paths(path1: Union[str, Path], path2: Union[str, Path]) -> Union[str, Path]:
    """
    Merges two paths intelligently by matching overlapping segments. Supports both str and pathlib.Path inputs.
    
    Args:
        path1: Base path (e.g., root directory). Can be a string or a pathlib.Path object.
        path2: Relative or absolute path to merge with base. Can be a string or a pathlib.Path object.
    
    Returns:
        Merged path as the same type as the input (str or pathlib.Path).
    
    Raises:
        ValueError: If no matching segments are found between the paths.
    """
    # Convert inputs to Path objects if they are strings
    path1_is_str = isinstance(path1, str)
    path2_is_str = isinstance(path2, str)
    path1 = Path(path1) if path1_is_str else path1
    path2 = Path(path2) if path2_is_str else path2

    # Normalize paths
    path1 = path1.resolve()
    path2 = Path(os.path.normpath(str(path2)))

    # List directories/files in path1
    try:
        path1_contents = set(p.name for p in path1.iterdir())
    except FileNotFoundError:
        raise ValueError(f"The base path '{path1}' does not exist.")

    # Split path2 into segments
    path2_segments = list(path2.parts)

    # Find matching segment in path1
    for i, segment in enumerate(path2_segments):
        if segment in path1_contents:
            matching_index = i
            break
    else:
        raise ValueError(f"No matching directory or file found in '{path1}' for '{path2}'.")

    # Merge paths starting from the matching segment in path2
    merged_path = path1.joinpath(*path2_segments[matching_index:])

    # Return the merged path in the same type as the input
    return str(merged_path) if path1_is_str else merged_path



SAGEMAKER_TRAINING_JOB_NAME_MAX_LEN = 39
def shorten_for_sagemaker_training_job(name: str) -> str:
    return name[:SAGEMAKER_TRAINING_JOB_NAME_MAX_LEN]


def log_progress(message: str, clear: bool = True):
    """Helper function to ensure immediate output in notebooks
    Args:
        message: Message to display
        clear: Whether to clear previous output (default: True)
    """
    logging.info(message)
    sys.stdout.flush()
    try:
        from IPython.display import clear_output, display
        if clear:
            clear_output(wait=True)
        display(message)
    except ImportError:
        pass  # Not in a notebook environment

### Deployment (resources) helpers 

from botocore.exceptions import ClientError

def check_model_exists(model_name, sm_client=None):
    if not sm_client:
        sm_client = boto3.client('sagemaker')
    try:
        sm_client.describe_model(ModelName=model_name)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            return False
        raise

def check_endpoint_config_exists(endpoint_config_name, sm_client=None):
    if not sm_client:
        sm_client = boto3.client('sagemaker')
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            return False
        raise

def check_endpoint_exists(endpoint_name, sm_client=None):
    if not sm_client:
        sm_client = boto3.client('sagemaker')
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' exists with status: {response['EndpointStatus']}.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"Endpoint '{endpoint_name}' does not exist.")
            return False
        raise

def delete_all_resources(model_name, endpoint_name, sm_client=None):
    if not sm_client:
        sm_client = boto3.client('sagemaker')
        
    if check_endpoint_exists(endpoint_name, sm_client):
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Deleted existing endpoint: {endpoint_name}")
            waiter = sm_client.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
        except ClientError as e:
            print(f"Error deleting endpoint: {e}")
    
    if check_endpoint_config_exists(endpoint_name, sm_client):
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            print(f"Deleted existing endpoint config: {endpoint_name}")
        except ClientError as e:
            print(f"Error deleting endpoint config: {e}")
    
    if check_model_exists(model_name, sm_client):
        try:
            sm_client.delete_model(ModelName=model_name)
            print(f"Deleted existing model: {model_name}")
        except ClientError as e:
            print(f"Error deleting model: {e}")

def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, "png")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def get_pil_image(im, dataset_base_dir=""):
    if type(im) is dict:
        im_bytes = im.get("bytes", None)
        if im_bytes:
            im = BytesIO(im_bytes)
        else:
            im = os.path.join(dataset_base_dir, im.get("path", None))

    image = Image.open(im)
    return image.resize((400, 600))

