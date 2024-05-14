from __future__ import annotations

import os
import sys
from datetime import datetime
from digit_classifier import DigitClassifier
from solver import SudokuSolver
from engine import ProcessorEngine

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.logger import Logger  # noqa
import utils.utils as utils  # noqa

if __name__ == "__main__":
    debug_mode = False
    image_file_path = r"F:\Git_Repos\SudokuSolverPipeline\sudoku_images\example_3.jpg"  # noqa #Tempopary
    logger = Logger(log_file=datetime.today().strftime("%Y_%m_%d") + "_sudoku_solver.log", debug_mode=debug_mode)

    classifier = DigitClassifier(model_path="dnet_classifier.pth", logger=logger)
    sudoku_preprocess_engine = ProcessorEngine(image=image_file_path, classifier=classifier, logger=logger)

    # GET BOARD
    sudoku_preprocess_engine.denoise_image(kernel_size=(7, 7))
    sudoku_preprocess_engine.apply_threshold()
    sudoku_preprocess_engine.dilate()
    sudoku_preprocess_engine.apply_perspective_transform()
    sudoku_preprocess_engine.pick_sudoku_cells()
    sudoku_preprocess_engine.get_digits()
    sudoku_board = sudoku_preprocess_engine.get_sudoku_board()

    solver = SudokuSolver(board=sudoku_board, logger=logger)
    solver.solve()
