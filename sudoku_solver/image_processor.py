from logger import Logger
from engine import ProcessorEngine

if __name__ == '__main__':

    debug_mode = True
    image_file_path = "F:\Git_Repos\SudokuSolverPipeline\sudoku_images\example_1.jpeg"
    logger = Logger(debug_mode=debug_mode)

    sudoku_preprocess_engine = ProcessorEngine(image_path=image_file_path, logger=logger, debug_mode=debug_mode)

    # GET BOARD
    sudoku_preprocess_engine.denoise_image(kernel_size=(7,7))
    sudoku_preprocess_engine.apply_threshold()
    sudoku_preprocess_engine.dilate()
    sudoku_preprocess_engine.apply_perspective_transform()


