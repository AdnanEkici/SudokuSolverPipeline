from __future__ import annotations

import os
import sys
from datetime import datetime

from engine import ProcessorEngine

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.logger import Logger  # noqa
import utils.utils as utils  # noqa

if __name__ == "__main__":
    debug_mode = True
    image_file_path = r"sudoku_images\example_1.jpeg"  # noqa #Tempopary
    logger = Logger(log_file=datetime.today().strftime("%Y_%m_%d") + "_sudoku_solver.log", debug_mode=debug_mode)

    sudoku_preprocess_engine = ProcessorEngine(image=image_file_path, logger=logger, debug_mode=debug_mode)

    # GET BOARD
    sudoku_preprocess_engine.denoise_image(kernel_size=(7, 7))
    sudoku_preprocess_engine.apply_threshold()
    sudoku_preprocess_engine.dilate()
    sudoku_preprocess_engine.apply_perspective_transform()
