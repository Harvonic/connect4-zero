"""
test_board.py

Basic tests for the Connect 4 Board environment.
"""

from board import Board


def test_empty_board():
    game = Board()

    assert game.board.shape == (6, 7)
    assert len(game.legal_moves()) == 7
    assert not game.gameOver
    assert game.winner is None


def test_piece_placement():
    game = Board()

    placed = game.place_piece(1, 3)

    assert placed
    assert game.board[5, 3] == 1
    assert game.colCount[3] == 1
    assert not game.gameOver


def test_pieces_stack_upward():
    game = Board()

    game.place_piece(1, 3)
    game.place_piece(-1, 3)

    assert game.board[5, 3] == 1
    assert game.board[4, 3] == -1
    assert game.colCount[3] == 2


def test_full_column_is_illegal():
    game = Board()

    for i in range(6):
        turn = 1 if i % 2 == 0 else -1
        assert game.place_piece(turn, 0)

    assert 0 not in game.legal_moves()
    assert not game.place_piece(1, 0)


def test_horizontal_win():
    game = Board()

    game.place_piece(1, 0)
    game.place_piece(1, 1)
    game.place_piece(1, 2)
    game.place_piece(1, 3)

    assert game.gameOver
    assert game.winner == 1


def test_vertical_win():
    game = Board()

    game.place_piece(-1, 2)
    game.place_piece(-1, 2)
    game.place_piece(-1, 2)
    game.place_piece(-1, 2)

    assert game.gameOver
    assert game.winner == -1


def test_bottom_left_to_top_right_diagonal():
    game = Board()

    # Build supports underneath the diagonal.
    game.place_piece(1, 0)

    game.place_piece(-1, 1)
    game.place_piece(1, 1)

    game.place_piece(-1, 2)
    game.place_piece(-1, 2)
    game.place_piece(1, 2)

    game.place_piece(-1, 3)
    game.place_piece(-1, 3)
    game.place_piece(-1, 3)
    game.place_piece(1, 3)

    assert game.gameOver
    assert game.winner == 1


def test_top_left_to_bottom_right_diagonal():
    game = Board()

    # Build supports underneath the diagonal.
    game.place_piece(-1, 0)
    game.place_piece(-1, 0)
    game.place_piece(-1, 0)
    game.place_piece(1, 0)

    game.place_piece(-1, 1)
    game.place_piece(-1, 1)
    game.place_piece(1, 1)

    game.place_piece(-1, 2)
    game.place_piece(1, 2)

    game.place_piece(1, 3)

    assert game.gameOver
    assert game.winner == 1


def test_reset():
    game = Board()

    game.place_piece(1, 0)
    game.reset()

    assert game.board.shape == (6, 7)
    assert game.board.sum() == 0
    assert game.colCount.sum() == 0
    assert len(game.legal_moves()) == 7
    assert not game.gameOver
    assert game.winner is None


def run_tests():
    tests = [
        test_empty_board,
        test_piece_placement,
        test_pieces_stack_upward,
        test_full_column_is_illegal,
        test_horizontal_win,
        test_vertical_win,
        test_bottom_left_to_top_right_diagonal,
        test_top_left_to_bottom_right_diagonal,
        test_reset,
    ]

    for test in tests:
        test()
        print(f"Passed: {test.__name__}")

    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()