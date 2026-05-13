"""SAC 主算法: actor + twin critic + 自动 alpha."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .networks import Actor, TwinCritic
from .config import SACConfig, NetworkConfig


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        sac_cfg: SACConfig,
        net_cfg: NetworkConfig,
        device: str = "cuda",
    ):
        self.cfg = sac_cfg
        self.device = device

        self.actor = Actor(
            state_dim,
            action_dim,
            net_cfg.hidden_sizes,
            action_low,
            action_high,
            net_cfg.log_std_min,
            net_cfg.log_std_max,
        ).to(device)
        self.critic = TwinCritic(state_dim, action_dim, net_cfg.hidden_sizes).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, net_cfg.hidden_sizes).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=sac_cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg.critic_lr)

        self.log_alpha = torch.tensor(
            np.log(sac_cfg.init_temperature), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=sac_cfg.alpha_lr)
        self.target_entropy = sac_cfg.target_entropy

        self.buffer = ReplayBuffer(state_dim, action_dim, sac_cfg.buffer_size, device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        action, _, eval_action = self.actor.sample(s)
        out = eval_action if deterministic else action
        return out.squeeze(0).cpu().numpy()

    def update(self) -> dict:
        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)

        # --- critic update ---
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(ns)
            tq1, tq2 = self.critic_target(ns, na)
            tq = torch.min(tq1, tq2) - self.alpha.detach() * nlp
            target = r + (1.0 - d) * self.cfg.gamma * tq

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor update ---
        na, lp, _ = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, na)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * lp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- alpha update ---
        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- soft target update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau)
                tp.data.add_(self.cfg.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q_pi.mean().item(),
            "logp_mean": lp.mean().item(),
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
