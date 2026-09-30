# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for replay-buffer based algorithms."""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import FlashSAC
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


class OffPolicyRunner(OnPolicyRunner):
    """Collect transitions, reset done environments, and update from replay."""

    alg: FlashSAC

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self.is_distributed:
            raise NotImplementedError("OffPolicyRunner does not support distributed training")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        update_credit = 0.0

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            loss_sums: dict[str, float] = {}
            loss_counts: dict[str, int] = {}

            for _ in range(self.cfg["num_steps_per_env"]):
                with torch.inference_mode():
                    actions = self.alg.act(obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                if self.cfg.get("check_for_nan", True):
                    check_nan(next_obs, rewards, dones)

                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                time_outs = extras.get("time_outs")
                truncated = (
                    time_outs.to(self.device, dtype=torch.bool)
                    if time_outs is not None
                    else torch.zeros_like(dones, dtype=torch.bool)
                )
                terminated = dones.bool() & ~truncated

                self.alg.process_transition(obs, actions, rewards, terminated, truncated, next_obs)

                step_extras = _copy_extras(extras)
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    with torch.inference_mode():
                        obs, reset_extras = self.reset_done(next_obs, done_ids)
                    step_extras = _merge_extras(step_extras, reset_extras)
                else:
                    obs = next_obs

                self.logger.process_env_step(rewards, dones, step_extras)

                if self.alg.can_update():
                    update_credit += self.cfg.get("updates_per_env_step", 1.0)
                    while update_credit >= 1.0:
                        for key, value in self.alg.update().items():
                            loss_sums[key] = loss_sums.get(key, 0.0) + value
                            loss_counts[key] = loss_counts.get(key, 0) + 1
                        update_credit -= 1.0

            collect_and_learn_time = time.time() - start
            loss_dict = {
                key: value / loss_counts[key] for key, value in loss_sums.items()
            }
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_and_learn_time,
                learn_time=0.0,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.action_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(  # type: ignore[arg-type]
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            )
            self.logger.stop_logging_writer()

    def reset_done(
        self, terminal_obs: TensorDict, env_ids: torch.Tensor
    ) -> tuple[TensorDict, dict]:
        """Reset completed environments and return observations for the next action."""
        raise NotImplementedError(
            "The environment adapter must implement reset_done() for OffPolicyRunner"
        )


def _copy_extras(extras: dict) -> dict:
    copied = dict(extras)
    for key in ("episode", "log"):
        if isinstance(copied.get(key), dict):
            copied[key] = dict(copied[key])
    return copied


def _merge_extras(step_extras: dict, reset_extras: dict) -> dict:
    merged = {**step_extras, **reset_extras}
    for key in ("episode", "log"):
        values = {}
        if isinstance(step_extras.get(key), dict):
            values.update(step_extras[key])
        if isinstance(reset_extras.get(key), dict):
            values.update(reset_extras[key])
        if values:
            merged[key] = values
    return merged
