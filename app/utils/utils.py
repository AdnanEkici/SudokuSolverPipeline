from __future__ import annotations

import cv2
import numpy as np
import yaml


def debug_show_image(image: str | np.ndarray, display_title: str = "DEBUG"):
    image = image
    if isinstance(image, str):
        image = cv2.imread(image)
    cv2.imshow(display_title, image)
    cv2.waitKey(0)


def calculate_euclidian_distance(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return np.sqrt((a**2) + (b**2))


def read_yaml(yaml_file_path, logger):
    logger.info(f"Reading yaml {yaml_file_path}")
    try:
        with open(yaml_file_path) as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        return yaml_data
    except FileNotFoundError:
        logger.critical(f"File '{yaml_file_path}' not found.")
        return None
    except Exception as e:
        logger.critical(f"Error reading YAML file '{yaml_file_path}': {e}")
        return None


def label_mapper(label: int):
    return label + 1 if label != 9 else "EMPTY"
