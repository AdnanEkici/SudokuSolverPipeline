from typing import Union
import cv2
import numpy as np


def debug_show_image(image: Union[str, np.ndarray], display_title: str = "DEBUG"):
    image = image
    if isinstance(image, str):
        image = cv2.imread(image)
    cv2.imshow(display_title, image)
    cv2.waitKey(0)


def calculate_euclidian_distance(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return np.sqrt((a ** 2) + (b ** 2))
