"""
train.py is responsible for handling the training pipeline
"""
import random

from replay import ReplayBuffer
from board import Board
from dqn import DQN
import cupy as cp
import numpy as np
import os



class Train:

    def __init__(self, batch_size=32, buffer_capacity=10000, gamma=0.99, epsilon=1.0):
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.gamma = gamma
        self.epsilon = epsilon

        self.buffer = ReplayBuffer(buffer_capacity)

        self.online = DQN(lr=0.001)
        self.target = DQN(lr=0.001)

        self.start_episode = 0
        self.training_steps = 0
        self.samples_used = 0
        self.samples_until_refresh = 100_000

        self.target.copy_weights_from(self.online)


    """
    Helper function which chooses the models action
    """
    def choose_network_action(self, game: Board, model: DQN, turn: int, epsilon: float = 0.0,):
        legal_moves = game.legal_moves()

        # Explore with probability epsilon.
        if random.random() < epsilon:
            return int(random.choice(legal_moves))

        flattened_board = (game.board * turn).reshape(42, 1) 

        input = cp.asarray(flattened_board, dtype=cp.float32)
        q_values = model.predict(input)

        # Convert to a simple CPU vector for legal-move selection.
        q_values_cpu = cp.asnumpy(q_values).reshape(7)

        # Prevent full columns from being selected.
        masked_q_values = np.full(7, -np.inf, dtype=np.float32)
        masked_q_values[legal_moves] = q_values_cpu[legal_moves]

        return int(np.argmax(masked_q_values))


    """
    Play game using DQN and add moves to buffer

    Returns the absolute winner:
            1, -1, or 0
    """
    def play_game(self):

        game = Board()
        turn = 1

        while not game.gameOver:
            state = (game.board * turn).reshape(42, 1).astype(np.float32)

            action = self.choose_network_action(game, self.online, turn=turn, epsilon=self.epsilon)

            game.place_piece(turn, action)
            done = game.gameOver

            if done and game.winner == turn:
                reward = 1.0
            else:
                reward = 0.0

            next_turn = -turn
            next_state = (game.board * next_turn).reshape(42, 1).astype(np.float32)

            mirrored_state = state.reshape(6, 7)[:, ::-1].reshape(42, 1)
            mirrored_next_state = next_state.reshape(6, 7)[:, ::-1].reshape(42, 1)
            mirrored_action = 6 - action

            self.buffer.add(
                mirrored_state.copy(),
                mirrored_action,
                reward,
                mirrored_next_state.copy(),
                done,
            )

            self.buffer.add(state.copy(), action, reward, next_state.copy(), done)

            if not done:
                turn = next_turn
        
        return game.winner

    """
    Fill the buffer with experiences
    """
    def fill_buffer(self):

        while len(self.buffer) < self.buffer_capacity:
            self.play_game()
    
    
    """
    Takes sample batches and returns:
    X          # (42, B)
    actions    # (B,)
    rewards    # (B,)
    next_X     # (42, B)
    dones      # (B,)
    """
    def sample_batches(self):
        samples = self.buffer.sample(self.batch_size)

    
        states, actions, rewards, next_states, dones = zip(*samples)

        X = cp.asarray(np.concatenate(states, axis=1), dtype=cp.float32)

        next_X = cp.asarray(np.concatenate(next_states, axis=1), dtype=cp.float32)

        actions = cp.asarray(actions, dtype=cp.int32)
        rewards = cp.asarray(rewards, dtype=cp.float32)
        dones = cp.asarray(dones, dtype=cp.bool_)

        return X, actions, rewards, next_X, dones

    """
    Samples one replay batch, computes Bellman targets,
    and updates the online network once.
    """
    def train_step(self):

        X, actions, rewards, next_X, dones = self.sample_batches()
        self.samples_used += self.batch_size

        if self.samples_used >= self.samples_until_refresh:
            self.buffer.empty()
            self.fill_buffer()
            self.samples_used = 0

        next_Q = self.target.predict(next_X)

        legal_mask = next_X[:7, :] == 0
        masked_next_Q = cp.where(legal_mask, next_Q, -cp.inf,)
        max_next_Q = cp.max(masked_next_Q, axis=0)


        targets = rewards.copy()
        non_terminal = ~dones

        targets[non_terminal] = (rewards[non_terminal]- self.gamma * max_next_Q[non_terminal])

        if not bool(cp.all(cp.isfinite(targets))):
            raise RuntimeError("Targets contain NaN or infinity.")

        loss = self.online.train_batch(X, actions, targets)

        return loss

    """
    Handle training the DQN
    """
    def train(self, episodes=10000, target_update_interval=500, epsilon_min=0.05, epsilon_decay=0.9995,):

        os.makedirs("checkpoints", exist_ok=True)

        self.target.copy_weights_from(self.online)
        self.fill_buffer()


        for episode in range(self.start_episode + 1, self.start_episode + episodes + 1):
            self.play_game()

            for _ in range(8):
                loss = self.train_step()
                self.training_steps += 1

                if self.training_steps % target_update_interval == 0:
                    self.target.copy_weights_from(self.online)

            # loss = self.train_step()
            # self.training_steps += 1

            # if self.training_steps % target_update_interval == 0:
            #     self.target.copy_weights_from(self.online)
            
            self.epsilon = max(
                epsilon_min,
                self.epsilon * epsilon_decay,
            )

            if episode % 100 == 0:
                print(
                    f"Episode: {episode} | "
                    f"Loss: {loss:.6f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                )

            if episode > 0 and episode % 1000 == 0:
                self.online.save_weights(
                    f"checkpoints/model_{episode}.npz",
                    episode=episode,
                    epsilon=self.epsilon,
                    training_steps=self.training_steps,
                )
        
        self.start_episode = episode

        self.online.save_weights(
            "dqn_weights.npz",
            episode=episode,
            epsilon=self.epsilon,
            training_steps=self.training_steps,
        )
    
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}. Starting from scratch.")
            return

        metadata = self.online.load_weights(path)

        self.start_episode = metadata.get("episode", 0)
        self.epsilon = metadata.get("epsilon", self.epsilon)
        self.training_steps = metadata.get("training_steps", 0)

        self.target.copy_weights_from(self.online)

trainer = Train(batch_size=128, buffer_capacity=1000)
trainer.load_checkpoint("dqn_weights.npz")

while True:
    trainer.train(episodes=30000, target_update_interval=20000)