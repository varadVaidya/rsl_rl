# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT

"""FlashSAC adapted to RSL-RL's TensorDict environment interface.

The update equations and network design follow Holiday-Robot/FlashSAC commit
87edc9061150ae9e962dd84e6544e27a1554b3ab. Environment interaction stays in
RSL-RL so simulator observations never make a GPU-to-NumPy round trip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.amp import GradScaler

from rsl_rl.env import VecEnv
from rsl_rl.modules.flash_sac import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
    normalize_parameters,
    update_ema,
)
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_obs_groups


def _flatten_observations(obs: TensorDict, groups: Sequence[str]) -> torch.Tensor:
    return torch.cat([obs[group] for group in groups], dim=-1)


def _cosine_schedule(
    initial: float, peak: float, final: float, warmup_steps: int, decay_steps: int
):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (initial + (peak - initial) * step / warmup_steps) / peak
        if step < decay_steps:
            progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
            value = final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return value / peak
        return final / peak

    return schedule


class _RunningMeanStd:
    def __init__(self, device: torch.device) -> None:
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = torch.tensor(0.0, device=device)

    def update(self, samples: torch.Tensor) -> None:
        sample_mean = samples.mean().reshape(1)
        sample_var = samples.var(unbiased=False).reshape(1)
        sample_count = float(samples.numel())
        delta = sample_mean - self.mean
        total = self.count + sample_count
        ratio = sample_count / total
        m2 = self.var * (self.count + 1e-4) + sample_var * sample_count
        m2 += delta.square() * self.count * ratio
        self.mean = self.mean + delta * ratio
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class _RewardNormalizer:
    def __init__(self, gamma: float, max_return: float, device: torch.device) -> None:
        self.gamma = gamma
        self.max_return = max_return
        self.return_estimate = torch.zeros(1, device=device)
        self.max_abs_return = torch.zeros(1, device=device)
        self.stats = _RunningMeanStd(device)

    def update(
        self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor
    ) -> None:
        done = terminated.bool() | truncated.bool()
        self.return_estimate = self.gamma * ~done * self.return_estimate + reward
        self.max_abs_return = torch.maximum(
            self.max_abs_return, self.return_estimate.abs().max().reshape(1)
        )
        self.stats.update(self.return_estimate)

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        denominator = torch.maximum(
            torch.sqrt(self.stats.var + 1e-8), self.max_abs_return / self.max_return
        )
        return reward / denominator

    def state_dict(self) -> dict:
        return {
            "return_estimate": self.return_estimate,
            "max_abs_return": self.max_abs_return,
            "stats": self.stats.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.return_estimate = state["return_estimate"]
        self.max_abs_return = state["max_abs_return"]
        self.stats.load_state_dict(state["stats"])


class FlashSACPolicy(nn.Module):
    """Deterministic actor view used by the existing play/evaluation code."""

    def __init__(self, actor: FlashSACActor, obs_groups: Sequence[str]) -> None:
        super().__init__()
        self.actor = actor
        self.obs_groups = tuple(obs_groups)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        mean, _ = self.actor.get_mean_and_std(
            _flatten_observations(obs, self.obs_groups), training=False
        )
        return torch.tanh(mean)


class FlashSAC:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        obs_groups: Sequence[str],
        buffer_max_length: int = 1_000_000,
        buffer_min_length: int = 100_000,
        sample_batch_size: int = 2048,
        gamma: float = 0.99,
        n_step: int = 3,
        normalize_reward: bool = True,
        normalized_return_max: float = 5.0,
        learning_rate_initial: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_final: float = 1.5e-4,
        learning_rate_warmup_steps: int = 100,
        learning_rate_decay_steps: int = 100_000,
        actor_num_blocks: int = 2,
        actor_hidden_dim: int = 128,
        actor_update_period: int = 2,
        actor_noise_zeta: float = 2.0,
        actor_noise_max_steps: int = 16,
        critic_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_bins: int = 101,
        critic_min_value: float = -5.0,
        critic_max_value: float = 5.0,
        critic_target_tau: float = 0.01,
        temperature_initial: float = 0.01,
        temperature_target_sigma: float = 0.15,
        use_amp: bool = True,
        save_replay_buffer: bool = False,
        device: str = "cpu",
        **_: object,
    ) -> None:
        self.device = torch.device(device)
        self.obs_groups = tuple(obs_groups)
        self.gamma = gamma
        self.n_step = n_step
        self.normalize_reward = normalize_reward
        self.actor_update_period = actor_update_period
        self.critic_num_bins = critic_num_bins
        self.critic_min_value = critic_min_value
        self.critic_max_value = critic_max_value
        self.critic_target_tau = critic_target_tau
        self.save_replay_buffer = save_replay_buffer
        self.update_step = 0

        self.actor = FlashSACActor(
            observation_dim, action_dim, actor_hidden_dim, actor_num_blocks
        ).to(self.device)
        self.critic = FlashSACDoubleCritic(
            observation_dim,
            action_dim,
            critic_hidden_dim,
            critic_num_blocks,
            critic_num_bins,
            critic_min_value,
            critic_max_value,
        ).to(self.device)
        self.target_critic = FlashSACDoubleCritic(
            observation_dim,
            action_dim,
            critic_hidden_dim,
            critic_num_blocks,
            critic_num_bins,
            critic_min_value,
            critic_max_value,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.temperature = FlashSACTemperature(temperature_initial).to(self.device)

        fused = self.device.type == "cuda"
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate_peak, fused=fused
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate_peak, fused=fused
        )
        self.temperature_optimizer = torch.optim.Adam(
            self.temperature.parameters(), lr=learning_rate_peak, fused=fused
        )
        schedule = _cosine_schedule(
            learning_rate_initial,
            learning_rate_peak,
            learning_rate_final,
            learning_rate_warmup_steps,
            learning_rate_decay_steps,
        )
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            for optimizer in (
                self.actor_optimizer,
                self.critic_optimizer,
                self.temperature_optimizer,
            )
        ]
        normalize_parameters(self.actor)
        normalize_parameters(self.critic)
        normalize_parameters(self.target_critic)

        self.replay_buffer = ReplayBuffer(
            observation_dim,
            action_dim,
            buffer_max_length,
            buffer_min_length,
            sample_batch_size,
            n_step,
            gamma,
            device,
        )
        self.reward_normalizer = (
            _RewardNormalizer(gamma, normalized_return_max, self.device)
            if normalize_reward
            else None
        )
        self.grad_scaler = GradScaler(
            self.device.type, enabled=use_amp and self.device.type == "cuda"
        )
        self.use_amp = use_amp and self.device.type == "cuda"

        probabilities = torch.arange(1, actor_noise_max_steps + 1, device=self.device) ** (
            -actor_noise_zeta
        )
        self.noise_cdf = (probabilities / probabilities.sum()).cumsum(0)
        self.noise = torch.randn(action_dim, device=self.device)
        self.noise_steps = 0
        self.noise_duration = 1
        self.action_std = torch.ones(action_dim, device=self.device)
        self.target_entropy = 0.5 * action_dim * math.log(
            2.0 * math.pi * math.e * temperature_target_sigma**2
        )

    @property
    def learning_rate(self) -> float:
        return float(self.actor_optimizer.param_groups[0]["lr"])

    def can_update(self) -> bool:
        return self.replay_buffer.can_sample()

    def act(self, obs: TensorDict, training: bool = True) -> torch.Tensor:
        flat_obs = _flatten_observations(obs, self.obs_groups).to(self.device)
        if training and not self.can_update():
            return torch.rand(flat_obs.shape[0], self.noise.shape[0], device=self.device) * 2 - 1
        with torch.no_grad():
            mean, std = self.actor.get_mean_and_std(flat_obs, training=False)
            self.action_std = std.mean(dim=0)
            if not training:
                return torch.tanh(mean)
            if self.noise_steps >= self.noise_duration:
                self.noise = torch.randn_like(mean)
                sample = torch.rand((), device=self.device)
                self.noise_duration = int(torch.searchsorted(self.noise_cdf, sample).item()) + 1
                self.noise_steps = 0
            elif self.noise.ndim == 1:
                self.noise = torch.randn_like(mean)
            self.noise_steps += 1
            return torch.tanh(mean + std * self.noise)

    def process_transition(
        self,
        obs: TensorDict,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: TensorDict,
    ) -> None:
        transition = {
            "observation": _flatten_observations(obs, self.obs_groups),
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": _flatten_observations(next_obs, self.obs_groups),
        }
        self.replay_buffer.add(transition)
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(reward, terminated, truncated)

    def update(self) -> dict[str, float]:
        batch = self.replay_buffer.sample()
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize(batch["reward"])

        metrics: dict[str, torch.Tensor] = {}
        if self.update_step % self.actor_update_period == 0:
            metrics.update(self._update_actor(batch))
            metrics.update(self._update_temperature(metrics["actor/entropy"]))
        metrics.update(self._update_critic(batch))
        update_ema(self.target_critic, self.critic, self.critic_target_tau)
        self.update_step += 1
        return {key: float(value.detach()) for key, value in metrics.items()}

    def _update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            observations = torch.cat(
                (batch["observation"], batch["next_observation"]), dim=0
            )
            actions, log_probs = self.actor(observations, training=True)
            actions = actions.chunk(2)[0]
            log_probs = log_probs.chunk(2)[0]
            self.critic.requires_grad_(False)
            q_values, _ = self.critic(batch["observation"], actions, training=False)
            self.critic.requires_grad_(True)
            q_value = torch.minimum(q_values[0], q_values[1])
            loss = (self.temperature().detach() * log_probs - q_value).mean()
            entropy = -log_probs.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.actor_optimizer)
        self.grad_scaler.update()
        self.schedulers[0].step()
        normalize_parameters(self.actor)
        return {
            "actor/loss": loss,
            "actor/entropy": entropy,
            "actor/mean_action": actions.mean(),
        }

    def _update_temperature(self, entropy: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.temperature()
        loss = value * (entropy.detach() - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.temperature_optimizer.step()
        self.schedulers[2].step()
        return {"temperature/value": value, "temperature/loss": loss}

    def _update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                next_actions, next_log_prob = self.actor(
                    batch["next_observation"], training=False
                )
                all_obs = torch.cat(
                    (batch["observation"], batch["next_observation"]), dim=0
                )
                all_actions = torch.cat((batch["action"], next_actions), dim=0)
                next_q, next_log_probs = self.target_critic(
                    all_obs, all_actions, training=True
                )
                next_q = next_q.chunk(2, dim=1)[1]
                next_log_probs = next_log_probs.chunk(2, dim=1)[1]
                indices = next_q.argmin(dim=0)
                indices = indices[None, :, None].expand(1, -1, self.critic_num_bins)
                selected_log_probs = torch.gather(next_log_probs, 0, indices)[0]
                target = self._categorical_target(
                    selected_log_probs,
                    batch["reward"],
                    batch["terminated"],
                    self.temperature() * next_log_prob,
                )

            _, predicted_log_probs = self.critic(all_obs, all_actions, training=True)
            predicted_log_probs = predicted_log_probs.chunk(2, dim=1)[0]
            loss = -(target.unsqueeze(0) * predicted_log_probs).sum(dim=-1).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.critic_optimizer)
        self.grad_scaler.update()
        self.schedulers[1].step()
        normalize_parameters(self.critic)
        return {
            "critic/loss": loss,
            "critic/max_entropy_bonus": (self.temperature() * next_log_prob).max(),
        }

    def _categorical_target(
        self,
        target_log_probs: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        entropy_bonus: torch.Tensor,
    ) -> torch.Tensor:
        width = (self.critic_max_value - self.critic_min_value) / (
            self.critic_num_bins - 1
        )
        bins = torch.linspace(
            self.critic_min_value,
            self.critic_max_value,
            self.critic_num_bins,
            device=self.device,
            dtype=target_log_probs.dtype,
        ).reshape(1, -1)
        target_bins = reward[:, None] + self.gamma**self.n_step * (
            bins - entropy_bonus[:, None]
        ) * (1.0 - terminated[:, None])
        target_bins.clamp_(self.critic_min_value, self.critic_max_value)
        projected = (target_bins - self.critic_min_value) / width
        lower = projected.floor().long()
        upper = (lower + 1).clamp(max=self.critic_num_bins - 1)
        fraction = projected - lower
        probabilities = target_log_probs.exp()
        target = torch.zeros_like(probabilities)
        target.scatter_add_(1, lower, probabilities * (1.0 - fraction))
        target.scatter_add_(1, upper, probabilities * fraction)
        return target

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_policy(self) -> FlashSACPolicy:
        return FlashSACPolicy(self.actor, self.obs_groups)

    def save(self) -> dict:
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "temperature_state_dict": self.temperature.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "temperature_optimizer_state_dict": self.temperature_optimizer.state_dict(),
            "scheduler_state_dicts": [scheduler.state_dict() for scheduler in self.schedulers],
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "update_step": self.update_step,
        }
        if self.reward_normalizer is not None:
            state["reward_normalizer_state_dict"] = self.reward_normalizer.state_dict()
        if self.save_replay_buffer:
            state["replay_buffer_state_dict"] = self.replay_buffer.state_dict()
        return state

    def load(self, state: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        load_cfg = load_cfg or {
            "actor": True,
            "critic": True,
            "optimizer": True,
            "normalizer": True,
            "replay_buffer": True,
            "iteration": True,
        }
        if load_cfg.get("actor"):
            self.actor.load_state_dict(state["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(state["critic_state_dict"], strict=strict)
            self.target_critic.load_state_dict(state["target_critic_state_dict"], strict=strict)
            self.temperature.load_state_dict(state["temperature_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
            self.temperature_optimizer.load_state_dict(state["temperature_optimizer_state_dict"])
            for scheduler, saved in zip(
                self.schedulers, state["scheduler_state_dicts"], strict=True
            ):
                scheduler.load_state_dict(saved)
            self.grad_scaler.load_state_dict(state["grad_scaler_state_dict"])
            self.update_step = int(state["update_step"])
        if load_cfg.get("normalizer") and self.reward_normalizer is not None:
            if saved := state.get("reward_normalizer_state_dict"):
                self.reward_normalizer.load_state_dict(saved)
        if load_cfg.get("replay_buffer") and (saved := state.get("replay_buffer_state_dict")):
            self.replay_buffer.load_state_dict(saved)
        return bool(load_cfg.get("iteration"))

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "FlashSAC":
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor"])
        groups = cfg["obs_groups"]["actor"]
        observation_dim = sum(obs[group].shape[-1] for group in groups)
        algorithm_cfg = dict(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        algorithm_cfg.pop("rnd_cfg", None)
        return FlashSAC(
            observation_dim=observation_dim,
            action_dim=env.num_actions,
            obs_groups=groups,
            device=device,
            **algorithm_cfg,
        )
