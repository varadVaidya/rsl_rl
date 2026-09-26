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
        phase: str | None = None,
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
            phase: Pipeline phase sharing the run. Inserted after the metric category
                (``Loss/value`` -> ``Loss/<phase>/value``) and prefixed to config keys.
            base_path: Root preserved when checkpoint and file paths are uploaded.

        Every metric is plotted against ``iteration`` (per phase, starting at 0);
        ``env_steps`` is logged alongside so the W&B x-axis can switch to it.
        """
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        super().__init__(log_dir, flush_secs=10)

        self._owns_run = wandb.run is None
        self.phase = phase
        self.base_path = base_path
        # One W&B history row per iteration, committed by flush().
        self._row: dict = {}

        if not self._owns_run:
            # A run was already started by the caller (e.g. a training launcher that needs
            # the generated run name to build ``log_dir`` before the runner exists). Reuse
            # it rather than starting a second run; just record the resolved log_dir.
            key = f"{self.phase}/log_dir" if self.phase else "log_dir"
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
            )

        # Phases restart at iteration 0, so W&B's monotonic _step cannot be the x-axis.
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

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
        """Log a scalar to TensorBoard and queue it for this iteration's W&B row."""
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        self._row[self._key(tag)] = scalar_value
        self._row["iteration"] = global_step

    def flush(self) -> None:
        """Commit the queued metrics as one W&B row."""
        super().flush()
        if self._row:
            wandb.log(self._row)
            self._row = {}

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        prefix = f"{self.phase}/" if self.phase else ""
        # A resumed run re-uploads its config with a different max_iterations.
        wandb.config.update({f"{prefix}train_cfg": train_cfg}, allow_val_change=True)
        try:
            wandb.config.update({f"{prefix}env_cfg": env_cfg.to_dict()}, allow_val_change=True)  # type: ignore
        except Exception:
            wandb.config.update({f"{prefix}env_cfg": asdict(env_cfg)}, allow_val_change=True)  # type: ignore

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=self.base_path or os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=self.base_path or os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Queue a video once per filename for this iteration's W&B row."""
        if video.name not in self.logged_videos:
            self._row[self._key("video")] = wandb.Video(str(video), format="mp4")
            self._row["iteration"] = it
            self.logged_videos.add(video.name)

    def stop(self) -> None:
        """Close this writer and finish only runs that it created."""
        self.flush()
        super().close()
        if self._owns_run:
            wandb.finish()

    def _key(self, tag: str) -> str:
        # env_steps is an x-axis shared by every phase, like iteration.
        if not self.phase or tag == "env_steps":
            return tag
        category, _, name = tag.partition("/")
        return f"{category}/{self.phase}/{name}" if name else f"{category}/{self.phase}"
