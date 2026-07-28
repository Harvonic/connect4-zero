"""
replay.py is responsible for handling the replay buffer
"""

import random
from collections import deque

class ReplayBuffer:

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )
    
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError(
                "Batch size cannot exceed replay-buffer size."
            )

        return random.sample(self.buffer, batch_size)
    
    def empty(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)