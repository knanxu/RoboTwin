"""SAC actor / twin critic.

动作直接输出绝对值: actor 出 tanh squashed gaussian, 再 linear map 到 [low, high]^3.
"""
import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: Sequence[int], activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed diagonal gaussian -> linear mapped to per-dim bounds."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.trunk = mlp([state_dim, *hidden_sizes])
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 记录 bounds 用于 tanh -> 实际 action 的 linear mapping
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

        # ReLU 在 trunk 最后一层后需要显式加一次
        self.post_relu = nn.ReLU()

    def _trunk_forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.trunk(s)
        return self.post_relu(h)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk_forward(s)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, tanh_mean_for_eval)."""
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # rsample for reparameterization
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        eval_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, eval_action


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = mlp([in_dim, *hidden_sizes, 1])
        self.q2 = mlp([in_dim, *hidden_sizes, 1])

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)
