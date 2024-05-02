from __future__ import annotations

from engine import ProcessorEngine
from logger import Logger
from datetime import datetime

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
