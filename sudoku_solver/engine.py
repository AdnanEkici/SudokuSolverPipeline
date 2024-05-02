from __future__ import annotations

import cv2
from logger import Logger
import utils
from typing import Tuple
import numpy as np
import operator
import cv2 as cv

class ProcessorEngine:
    def __init__(self, image_path: str, logger: Logger, debug_mode:bool = False):

        self.logger = logger
        self.sudoku_image = cv2.imread(image_path, 0)
        self.debug_show_image = utils.debug_show_image if debug_mode else None
        self.logger.debug(f"Reading image {image_path}")
        self.debug_show_image(image=self.sudoku_image)



    def denoise_image(self, kernel_size: Tuple[int, int] = (7, 7)):
        self.logger.debug(f"Applying gaussian blur with kernel: {kernel_size}")
        self.sudoku_image = cv2.GaussianBlur(self.sudoku_image.copy(), kernel_size, 0)
        self.debug_show_image(display_title="Blurred Image", image=self.sudoku_image)



    def apply_threshold(self):
        self.sudoku_image = cv2.adaptiveThreshold(self.sudoku_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)# Adaptive threshold using 11 nearest neighbour pixels
        self.sudoku_image = cv.bitwise_not(self.sudoku_image, self.sudoku_image)
        self.debug_show_image(display_title="Thresholded Image", image=self.sudoku_image)


    def dilate(self):
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        # self.sudoku_image = cv2.dilate(self.sudoku_image, kernel)
        self.sudoku_image = cv.morphologyEx(self.sudoku_image, cv.MORPH_OPEN, kernel, iterations=1)
        # self.debug_show_image(display_title="Dilated Image", image=self.sudoku_image)
        self.debug_show_image(display_title="Opening", image=self.sudoku_image)

    def get_board_corners(self):
        contours, h = cv.findContours(self.sudoku_image.copy(), cv2.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
        contours = sorted(contours, key=cv.contourArea, reverse=True)  # Sort by area, descending
        polygon = contours[0]  # Largest image

        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        corner_debug = cv.cvtColor(self.sudoku_image, cv.COLOR_GRAY2RGB)
        corner_debug = cv.circle(corner_debug, (polygon[top_left][0]), radius=1, color=(0, 255, 0), thickness=8)
        corner_debug = cv.circle(corner_debug, (polygon[top_right][0]), radius=1, color=(0, 0, 255), thickness=8)
        corner_debug = cv.circle(corner_debug, (polygon[bottom_right][0]), radius=1, color=(255, 0, 0), thickness=8)
        corner_debug = cv.circle(corner_debug, (polygon[bottom_left][0]), radius=1, color=(255, 255, 0), thickness=8)
        self.debug_show_image(display_title="Corners Of Game Field" , image=corner_debug)
        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



    def apply_perspective_transform(self):
        top_left, top_right, bottom_right, bottom_left = self.get_board_corners()
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        side = max([
            utils.calculate_euclidian_distance(bottom_right, top_right),
            utils.calculate_euclidian_distance(top_left, bottom_left),
            utils.calculate_euclidian_distance(bottom_right, bottom_left),
            utils.calculate_euclidian_distance(top_left, top_right)
        ])
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(src, dst)
        self.sudoku_image = cv2.warpPerspective(self.sudoku_image, m, (int(side), int(side)))
        self.debug_show_image(display_title="Transformed Image", image=self.sudoku_image)
        return self.sudoku_image