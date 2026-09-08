# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.utils.log_writer import LogWriter

try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None


class WandbLogWriter(SummaryWriter, LogWriter):
    """Summary writer for W&B."""

    def __init__(
        self,
        log_dir: str,
        project_name: str,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        notes: str | None = None,
        entity: str | None = None,
        wandb_dir: str | None = None,
        metric_namespace: str | None = None,
        base_path: str | None = None,
    ) -> None:
        """Initialize a W&B run for logging.

        Args:
            log_dir: The rsl_rl run directory (checkpoints, tensorboard, videos).
            project_name: The W&B project.
            run_name: W&B run display name. If ``None`` (default), W&B auto-generates
                its own random name (e.g. "stellar-dawn-7") rather than reusing the
                ``log_dir`` basename.
            group: W&B run group, for grouping related runs together.
            tags: W&B run tags.
            notes: Free-text notes attached to the W&B run.
            entity: W&B entity (team/user). Falls back to ``$WANDB_ENTITY`` then
                ``$WANDB_USERNAME`` when ``None``.
            wandb_dir: Directory for W&B's own local run files (``wandb.init(dir=...)``).
                Independent of ``log_dir``.
            metric_namespace: Optional prefix for every metric and config key.
            base_path: Root preserved when checkpoint and file paths are uploaded.
        """
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        super().__init__(log_dir, flush_secs=10)

        self._owns_run = wandb.run is None
        self.metric_namespace = metric_namespace.rstrip("/") if metric_namespace else None
        self.base_path = base_path

        if not self._owns_run:
            # A run was already started by the caller (e.g. a training launcher that needs
            # the generated run name to build ``log_dir`` before the runner exists). Reuse
            # it rather than starting a second run; just record the resolved log_dir.
            key = f"{self.metric_namespace}/log_dir" if self.metric_namespace else "log_dir"
            wandb.config.update({key: log_dir}, allow_val_change=True)
        else:
            if entity is None:
                entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME")
            # name=None lets W&B pick its default random run name.
            wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                group=group,
                tags=list(tags) if tags else None,
                notes=notes,
                dir=wandb_dir,
                config={"log_dir": log_dir},
                settings=wandb.Settings(start_method="thread"),
            )

        if self.metric_namespace:
            step_metric = f"{self.metric_namespace}/iteration"
            wandb.define_metric(step_metric)
            wandb.define_metric(f"{self.metric_namespace}/*", step_metric=step_metric)

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        if self.metric_namespace:
            wandb.log({
                f"{self.metric_namespace}/{tag}": scalar_value,
                f"{self.metric_namespace}/iteration": global_step,
            })
        else:
            wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        prefix = f"{self.metric_namespace}/" if self.metric_namespace else ""
        allow_change = self.metric_namespace is not None
        wandb.config.update({f"{prefix}train_cfg": train_cfg}, allow_val_change=allow_change)
        try:
            wandb.config.update(
                {f"{prefix}env_cfg": env_cfg.to_dict()},  # type: ignore
                allow_val_change=allow_change,
            )
        except Exception:
            wandb.config.update(
                {f"{prefix}env_cfg": asdict(env_cfg)},  # type: ignore
                allow_val_change=allow_change,
            )

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=self.base_path or os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=self.base_path or os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            tag = f"{self.metric_namespace}/video" if self.metric_namespace else "video"
            metrics = {tag: wandb.Video(str(video), format="mp4")}
            if self.metric_namespace:
                metrics[f"{self.metric_namespace}/iteration"] = it
                wandb.log(metrics)
            else:
                wandb.log(metrics, step=it)
            self.logged_videos.add(video.name)

    def stop(self) -> None:
        """Close this writer and finish only runs that it created."""
        super().close()
        if self._owns_run:
            wandb.finish()
