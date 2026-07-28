"""
main.py

Runs Connect 4 games for testing:

- Random vs random
- Network vs random
"""

import os
import random

import cupy as cp
import numpy as np

from board import Board
from dqn import DQN


def choose_random_action(game: Board) -> int:
    """Returns a random legal action."""
    legal_moves = game.legal_moves()
    return int(random.choice(legal_moves))


def choose_network_action(
    game: Board,
    model: DQN,
    turn: int,
) -> int:
    """
    Chooses the legal move with the highest predicted Q-value.

    The board is converted to the current player's perspective:
        current player = 1
        opponent = -1
    """
    legal_moves = game.legal_moves()

    # Board is stored using absolute player identities.
    # Multiplying by turn shows it from the current player's perspective.
    state_cpu = (game.board * turn).reshape(42, 1)

    # Move the state to the GPU.
    state = cp.asarray(state_cpu, dtype=cp.float32)

    # Shape: (7, 1)
    q_values = model.predict(state)

    # Convert to a simple CPU vector for legal-move selection.
    q_values_cpu = cp.asnumpy(q_values).reshape(7)

    # Prevent full columns from being selected.
    masked_q_values = np.full(7, -np.inf, dtype=np.float32)
    masked_q_values[legal_moves] = q_values_cpu[legal_moves]

    return int(np.argmax(masked_q_values))


def play_random_game(print_result: bool = True) -> int:
    """
    Plays one random-vs-random game.

    Returns:
        1  if Player 1 wins
        -1 if Player 2 wins
        0  if the game ends in a draw
    """
    game = Board()
    turn = 1

    while not game.gameOver:
        action = choose_random_action(game)

        if not game.place_piece(turn, action):
            raise RuntimeError(
                f"Failed to place a piece in legal column {action}."
            )

        if not game.gameOver:
            turn *= -1

    if print_result:
        game.print_board()

        if game.winner == 1:
            print("Player 1 wins!")
        elif game.winner == -1:
            print("Player 2 wins!")
        else:
            print("Draw!")

    return game.winner


def run_random_games(number_of_games: int) -> None:
    """Runs several random-vs-random games."""
    results = {
        1: 0,
        -1: 0,
        0: 0,
    }

    for _ in range(number_of_games):
        winner = play_random_game(print_result=False)
        results[winner] += 1

    print("\nRandom vs random")
    print(f"Games played: {number_of_games}")
    print(f"Player 1 wins: {results[1]}")
    print(f"Player 2 wins: {results[-1]}")
    print(f"Draws: {results[0]}")


def play_network_vs_random(
    model: DQN,
    network_player: int = 1,
    print_result: bool = True,
) -> int:
    """
    Plays one game between the network and a random player.

    Args:
        model:
            The DQN used by the network player.

        network_player:
            1 means the network is Player 1.
            -1 means the network is Player 2.

    Returns:
        1  if the network wins
        -1 if the network loses
        0  if the game is a draw
    """
    if network_player not in (1, -1):
        raise ValueError("network_player must be either 1 or -1.")

    game = Board()
    turn = 1

    while not game.gameOver:
        if turn == network_player:
            action = choose_network_action(game, model, turn)
        else:
            action = choose_random_action(game)

        if not game.place_piece(turn, action):
            raise RuntimeError(
                f"Failed to place a piece in legal column {action}."
            )

        if not game.gameOver:
            turn *= -1

    if game.winner == 0:
        network_result = 0
    elif game.winner == network_player:
        network_result = 1
    else:
        network_result = -1

    if print_result:
        game.print_board()

        print(
            "Network played as:",
            "Player 1" if network_player == 1 else "Player 2",
        )

        if network_result == 1:
            print("Network wins!")
        elif network_result == -1:
            print("Random player wins!")
        else:
            print("Draw!")

    return network_result


def run_network_vs_random_games(
    model: DQN,
    number_of_games: int,
) -> None:
    """
    Runs network-vs-random games and tracks results.

    The network alternates between Player 1 and Player 2 so that the
    first-player advantage does not distort the result as much.
    """
    network_wins = 0
    random_wins = 0
    draws = 0

    network_as_player_1_wins = 0
    network_as_player_2_wins = 0

    for game_index in range(number_of_games):
        network_player = 1 if game_index % 2 == 0 else -1

        result = play_network_vs_random(
            model=model,
            network_player=network_player,
            print_result=False,
        )

        if result == 1:
            network_wins += 1

            if network_player == 1:
                network_as_player_1_wins += 1
            else:
                network_as_player_2_wins += 1

        elif result == -1:
            random_wins += 1

        else:
            draws += 1

    print("\nNetwork vs random")
    print(f"Games played: {number_of_games}")
    print(f"Network wins: {network_wins}")
    print(f"Random wins: {random_wins}")
    print(f"Draws: {draws}")
    print(
        f"Network wins as Player 1: "
        f"{network_as_player_1_wins}"
    )
    print(
        f"Network wins as Player 2: "
        f"{network_as_player_2_wins}"
    )
    print(
        f"Network win rate: "
        f"{100 * network_wins / number_of_games:.2f}%"
    )


if __name__ == "__main__":
    model = DQN(lr=0.0003)
    
    weights_file = "dqn_weights.npz"

    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print("Testing saved network weights.")
    else:
        print("No saved weights found. Testing an untrained network.")

    run_network_vs_random_games(
        model=model,
        number_of_games=1000,
    )