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
        """
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        super().__init__(log_dir, flush_secs=10)

        if wandb.run is not None:
            # A run was already started by the caller (e.g. a training launcher that needs
            # the generated run name to build ``log_dir`` before the runner exists). Reuse
            # it rather than starting a second run; just record the resolved log_dir.
            wandb.config.update({"log_dir": log_dir}, allow_val_change=True)
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
        wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        wandb.config.update({"train_cfg": train_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})  # type: ignore
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})  # type: ignore

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)

    def stop(self) -> None:
        """Finish the active W&B run."""
        wandb.finish()
