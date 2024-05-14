class SudokuSolver:
    def __init__(self, board, logger):
        self.board = board
        self.size = 9
        self.logger = logger
        self.iterations = 0

    def __is_valid(self, num, pos):
        # Check row
        for col in range(self.size):
            if self.board[pos[0]][col] == num:
                return False

        # Check column
        for row in range(self.size):
            if self.board[row][pos[1]] == num:
                return False

        # Check 3x3 box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for row in range(box_y * 3, box_y * 3 + 3):
            for col in range(box_x * 3, box_x * 3 + 3):
                if self.board[row][col] == num:
                    return False

        return True

    def __find_empty(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    return (row, col)  # row, col
        return None

    def __solve(self):
        find = self.__find_empty()
        if not find:
            return True
        else:
            row, col = find

        for num in range(1, self.size + 1):
            self.iterations += 1
            if self.__is_valid(num, (row, col)):
                self.board[row][col] = num

                if self.__solve():
                    return True

                self.board[row][col] = 0
        return False

    def __show_board(self):
        board_str = ""
        for row in range(self.size):
            if row % 3 == 0 and row != 0:
                board_str += "- - - - - - - - - - - - \n"

            for col in range(self.size):
                if col % 3 == 0 and col != 0:
                    board_str += " | "

                if col == 8:
                    board_str += str(self.board[row][col]) + "\n"
                else:
                    board_str += str(self.board[row][col]) + " "
        self.logger.info("Board: \n" + board_str)

    def solve(self):
        self.__show_board()
        is_solved = self.__solve()
        self.__show_board()
        self.logger.info(f"This sudoku puzzle have been solved with {self.iterations} iterations")
        self.logger.warning("No solution exists for the given Sudoku puzzle.") if not is_solved else None

