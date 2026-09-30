# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT

"""GPU replay buffer adapted from Holiday-Robot/FlashSAC."""

from __future__ import annotations

from collections import deque

import torch


class ReplayBuffer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_length: int,
        min_length: int,
        batch_size: int,
        n_step: int,
        gamma: float,
        device: str,
    ) -> None:
        self.max_length = max_length
        self.min_length = min_length
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = torch.device(device)
        self.observations = torch.empty(max_length, observation_dim, device=self.device)
        self.next_observations = torch.empty(max_length, observation_dim, device=self.device)
        self.actions = torch.empty(max_length, action_dim, device=self.device)
        self.rewards = torch.empty(max_length, device=self.device)
        self.terminated = torch.empty(max_length, device=self.device)
        self.truncated = torch.empty(max_length, device=self.device)
        self.pending: deque[dict[str, torch.Tensor]] = deque(maxlen=n_step)
        self.length = 0
        self.index = 0

    def can_sample(self) -> bool:
        return self.length >= self.min_length

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        self.pending.append(
            {key: value.detach().to(self.device, copy=True) for key, value in transition.items()}
        )
        if len(self.pending) < self.n_step:
            return

        first = self.pending[0]
        latest = self.pending[-1]
        reward = latest["reward"].clone()
        terminated = latest["terminated"].clone()
        truncated = latest["truncated"].clone()
        next_observation = latest["next_observation"].clone()
        for transition in reversed(tuple(self.pending)[:-1]):
            done = transition["terminated"].bool() | transition["truncated"].bool()
            reward = transition["reward"] + self.gamma * reward * ~done
            terminated[done] = transition["terminated"][done]
            truncated[done] = transition["truncated"][done]
            next_observation[done] = transition["next_observation"][done]

        count = len(first["observation"])
        indices = (torch.arange(count, device=self.device) + self.index) % self.max_length
        self.observations[indices] = first["observation"]
        self.actions[indices] = first["action"]
        self.rewards[indices] = reward
        self.terminated[indices] = terminated.to(self.terminated.dtype)
        self.truncated[indices] = truncated.to(self.truncated.dtype)
        self.next_observations[indices] = next_observation
        self.length = min(self.length + count, self.max_length)
        self.index = (self.index + count) % self.max_length

    def sample(self) -> dict[str, torch.Tensor]:
        indices = torch.randint(self.length, (self.batch_size,), device=self.device)
        return {
            "observation": self.observations[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
            "next_observation": self.next_observations[indices],
        }

    def state_dict(self) -> dict:
        return {
            "observation": self.observations[: self.length],
            "next_observation": self.next_observations[: self.length],
            "action": self.actions[: self.length],
            "reward": self.rewards[: self.length],
            "terminated": self.terminated[: self.length],
            "truncated": self.truncated[: self.length],
            "length": self.length,
            "index": self.index,
        }

    def load_state_dict(self, state: dict) -> None:
        length = int(state["length"])
        if length > self.max_length:
            raise ValueError(f"replay checkpoint has {length} entries; capacity is {self.max_length}")
        self.observations[:length] = state["observation"].to(self.device)
        self.next_observations[:length] = state["next_observation"].to(self.device)
        self.actions[:length] = state["action"].to(self.device)
        self.rewards[:length] = state["reward"].to(self.device)
        self.terminated[:length] = state["terminated"].to(self.device)
        self.truncated[:length] = state["truncated"].to(self.device)
        self.length = length
        self.index = int(state["index"])
        self.pending.clear()
