"""
play.py lets a human play Connect Four against a trained DQN using Pygame.

Expected project files:
    board.py
    dqn.py
    checkpoint.npz   (or another checkpoint path)

Controls:
    - Click a column to place a piece.
    - Press R to restart.
    - Press ESC or close the window to quit.
"""

import os
import sys
import random

import cupy as cp
import numpy as np
import pygame

from board import Board
from dqn import DQN


# -----------------------------
# Configuration
# -----------------------------

WIDTH = 700
HEIGHT = 700
ROWS = 6
COLS = 7
CELL_SIZE = WIDTH // COLS
TOP_MARGIN = HEIGHT - ROWS * CELL_SIZE

FPS = 60

CHECKPOINT_PATH = "dqn_weights.npz"

# Human can be 1 or -1.
# Player 1 moves first.
HUMAN_PLAYER = 1
MODEL_PLAYER = -1


# -----------------------------
# Helpers
# -----------------------------

def choose_model_action(game: Board, model: DQN, turn: int) -> int:
    """
    Choose the model's greedy legal action.

    The board is multiplied by turn so the model always sees itself
    as Player 1, matching the training setup.
    """
    legal_moves = game.legal_moves()

    if legal_moves.size == 0:
        raise RuntimeError("No legal moves are available.")

    state = (game.board * turn).reshape(42, 1)
    state_gpu = cp.asarray(state, dtype=cp.float32)

    q_values = model.predict(state_gpu)
    q_values_cpu = cp.asnumpy(q_values).reshape(7)

    masked_q_values = np.full(7, -np.inf, dtype=np.float32)
    masked_q_values[legal_moves] = q_values_cpu[legal_moves]

    return int(np.argmax(masked_q_values))


def board_cell_is_empty(value) -> bool:
    return int(value) == 0


def draw_game(
    screen: pygame.Surface,
    game: Board,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    hover_column: int | None,
    current_turn: int,
) -> None:
    screen.fill((245, 245, 245))

    # Top status area.
    if game.gameOver:
        if game.winner == HUMAN_PLAYER:
            status = "You win! Press R to restart."
        elif game.winner == MODEL_PLAYER:
            status = "Model wins. Press R to restart."
        else:
            status = "Draw. Press R to restart."
    elif current_turn == HUMAN_PLAYER:
        status = "Your turn"
    else:
        status = "Model thinking..."

    text = font.render(status, True, (20, 20, 20))
    screen.blit(text, text.get_rect(center=(WIDTH // 2, TOP_MARGIN // 2)))

    controls = small_font.render(
        "Click a column | R: restart | Esc: quit",
        True,
        (70, 70, 70),
    )
    screen.blit(
        controls,
        controls.get_rect(center=(WIDTH // 2, TOP_MARGIN - 18)),
    )

    # Hover preview.
    if (
        not game.gameOver
        and current_turn == HUMAN_PLAYER
        and hover_column is not None
        and hover_column in game.legal_moves()
    ):
        preview_x = hover_column * CELL_SIZE + CELL_SIZE // 2
        preview_y = TOP_MARGIN // 2
        pygame.draw.circle(
            screen,
            (220, 60, 60),
            (preview_x, preview_y),
            CELL_SIZE // 2 - 10,
        )

    # Draw board background.
    board_rect = pygame.Rect(
        0,
        TOP_MARGIN,
        COLS * CELL_SIZE,
        ROWS * CELL_SIZE,
    )
    pygame.draw.rect(screen, (35, 90, 210), board_rect)

    # Draw cells.
    for row in range(ROWS):
        for col in range(COLS):
            value = int(game.board[row, col])

            center_x = col * CELL_SIZE + CELL_SIZE // 2
            center_y = TOP_MARGIN + row * CELL_SIZE + CELL_SIZE // 2

            if board_cell_is_empty(value):
                color = (245, 245, 245)
            elif value == 1:
                color = (220, 60, 60)
            else:
                color = (245, 210, 50)

            pygame.draw.circle(
                screen,
                color,
                (center_x, center_y),
                CELL_SIZE // 2 - 8,
            )

    pygame.display.flip()


def restart_game() -> tuple[Board, int]:
    return Board(), 1


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Connect Four vs DQN")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)

    model = DQN(lr=0.001)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "The game will use an untrained network."
        )

    game, current_turn = restart_game()
    hover_column = None
    running = True

    # If the model is Player 1, it should make the opening move.
    model_move_pending = current_turn == MODEL_PLAYER

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    game, current_turn = restart_game()
                    model_move_pending = current_turn == MODEL_PLAYER

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, _ = event.pos
                hover_column = mouse_x // CELL_SIZE

                if not 0 <= hover_column < COLS:
                    hover_column = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (
                    event.button == 1
                    and not game.gameOver
                    and current_turn == HUMAN_PLAYER
                ):
                    mouse_x, _ = event.pos
                    column = mouse_x // CELL_SIZE

                    if column in game.legal_moves():
                        game.place_piece(HUMAN_PLAYER, int(column))

                        if not game.gameOver:
                            current_turn = MODEL_PLAYER
                            model_move_pending = True

        # Make the model move outside the event loop.
        if (
            running
            and model_move_pending
            and not game.gameOver
            and current_turn == MODEL_PLAYER
        ):
            draw_game(
                screen,
                game,
                font,
                small_font,
                hover_column,
                current_turn,
            )

            pygame.time.delay(250)

            action = choose_model_action(
                game,
                model,
                MODEL_PLAYER,
            )
            game.place_piece(MODEL_PLAYER, action)

            model_move_pending = False

            if not game.gameOver:
                current_turn = HUMAN_PLAYER

        draw_game(
            screen,
            game,
            font,
            small_font,
            hover_column,
            current_turn,
        )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()