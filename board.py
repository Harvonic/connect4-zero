"""
board.py has all the logic regarding the connect 4 board
"""

import numpy as np

class Board:

    def __init__(self):
        self.gameOver = False
        self.board = np.zeros((6,7), dtype=int)
        self.colCount = np.zeros(7, dtype=int)
        self.winner = None

    def reset(self):
        self.gameOver = False
        self.winner = None
        self.board = np.zeros((6, 7), dtype=np.int8)
        self.colCount = np.zeros(7, dtype=np.int8)
    
    """
    Returns indices of legal moves
    """
    def legal_moves(self):
        return np.where(self.colCount < 6)[0]
    
    """
    Checks if there is a "connect 4" for a point

    Returns True if theres a connect 4, False otherwise
    """
    def check_win(self, turn, lastRow, column):

        def in_bounds(row, col):
            return (0 <= row < 6) and (0 <= col < 7)

        # check downwards

        contiguous = 0
        row, col = lastRow, column
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            row += 1

        if contiguous >= 4:
            return True

        # check sides

        contiguous = 0
        row, col = lastRow, column

        # check left one first
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            col -= 1

        # check right
        col = column + 1
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            col += 1

        if contiguous >= 4:
            return True

        # check diag (top left to bottom right)

        contiguous = 0
        row, col = lastRow, column

        # check top left first
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            row -= 1
            col -= 1

        # check bottom right
        row, col = lastRow + 1, column + 1
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            row += 1
            col += 1

        if contiguous >= 4:
            return True

        # check diag (bottom left to top right)

        contiguous = 0
        row, col = lastRow, column

        # check bottom left first
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            row += 1
            col -= 1

        # check top right
        row, col = lastRow - 1, column + 1
        while (in_bounds(row, col) and self.board[row][col] == turn):
            contiguous += 1
            row -= 1
            col += 1

        if contiguous >= 4:
            return True

        return False
    

    """
    Places a piece on the board, corresponding with the turn
    """
    def place_piece(self, turn, column):
        if self.gameOver:
            return False

        if column not in self.legal_moves():
            return False

        row = 5 - self.colCount[column]

        self.board[row, column] = turn
        self.colCount[column] += 1

        if self.check_win(turn, row, column):
            self.gameOver = True
            self.winner = turn
        elif len(self.legal_moves()) == 0:
            self.gameOver = True
            self.winner = 0

        return True
        

    def print_board(self):
        print(self.board)