from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, call

from pytest import MonkeyPatch

from rsl_rl.utils import wandb_log_writer


@dataclass
class _EnvConfig:
    num_envs: int = 4


def test_phase_writer_logs_one_row_per_iteration(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Phase metrics share one row per iteration and never finish the launcher's run."""
    fake_wandb = Mock()
    fake_wandb.run = object()
    fake_wandb.config = Mock()
    monkeypatch.setattr(wandb_log_writer, "wandb", fake_wandb)

    writer = wandb_log_writer.WandbLogWriter(
        str(tmp_path / "phase"),
        "project",
        phase="teacher",
        base_path=str(tmp_path),
    )
    writer.add_scalar("Loss/value", 1.25, global_step=7)
    writer.add_scalar("Train/mean_reward", 3.5, global_step=7)
    writer.add_scalar("env_steps", 8192, global_step=7)
    fake_wandb.log.assert_not_called()
    writer.flush()
    writer.store_config(_EnvConfig(), {"seed": 42})
    writer.save_model(str(tmp_path / "phase" / "model_7.pt"), 7)
    writer.stop()

    assert fake_wandb.define_metric.call_args_list == [
        call("iteration"),
        call("*", step_metric="iteration"),
    ]
    fake_wandb.log.assert_called_once_with({
        "Loss/teacher/value": 1.25,
        "Train/teacher/mean_reward": 3.5,
        "env_steps": 8192,
        "iteration": 7,
    })
    assert call({"teacher/train_cfg": {"seed": 42}}, allow_val_change=True) in fake_wandb.config.update.call_args_list
    fake_wandb.save.assert_called_once_with(str(tmp_path / "phase" / "model_7.pt"), base_path=str(tmp_path))
    fake_wandb.finish.assert_not_called()
