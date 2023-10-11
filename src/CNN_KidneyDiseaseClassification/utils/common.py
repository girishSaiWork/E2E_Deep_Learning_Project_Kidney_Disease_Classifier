import os
from box.exceptions import BoxValueError
import yaml
from CNN_KidneyDiseaseClassification import logger  # Assuming you have a logger defined in your module
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

# Function to read a YAML file and return its content as a ConfigBox object
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)  # Load YAML content
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)  # Return as a ConfigBox for easy access
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

# Function to create directories from a list of paths
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)  # Create the directory, if it doesn't exist
        if verbose:
            logger.info(f"created directory at: {path}")

# Function to save data as a JSON file
@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)  # Save data as JSON with indentation for readability
    logger.info(f"json file saved at: {path}")

# Function to load data from a JSON file and return it as a ConfigBox
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)  # Load JSON data
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)  # Return as a ConfigBox

# Function to save data as a binary file using joblib
@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)  # Save data as a binary file using joblib
    logger.info(f"binary file saved at: {path}")

# Function to load data from a binary file using joblib
@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)  # Load binary data using joblib
    logger.info(f"binary file loaded from: {path}")
    return data

# Function to get the size of a file in KB
@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)  # Calculate the size in KB
    return f"~ {size_in_kb} KB"

# Function to decode a base64 encoded image string and save it to a file
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)  # Decode base64 string
    with open(fileName, 'wb') as f:
        f.write(imgdata)  # Write decoded data to a file

# Function to encode an image into a base64 string
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())  # Read and encode image as base64
