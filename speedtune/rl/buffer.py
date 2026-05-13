"""轻量 numpy replay buffer, 配 SAC 使用."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.from_numpy(x).to(self.device)
        return (
            to_t(self.states[idx]),
            to_t(self.actions[idx]),
            to_t(self.rewards[idx]),
            to_t(self.next_states[idx]),
            to_t(self.dones[idx]),
        )

    def __len__(self):
        return self.size
