from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, call

from pytest import MonkeyPatch

from rsl_rl.utils import wandb_log_writer


@dataclass
class _EnvConfig:
    num_envs: int = 4


def test_phase_writer_reuses_run_and_namespaces_outputs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A phase writer must not finish its launcher's shared W&B run."""
    fake_wandb = Mock()
    fake_wandb.run = object()
    fake_wandb.config = Mock()
    monkeypatch.setattr(wandb_log_writer, "wandb", fake_wandb)

    writer = wandb_log_writer.WandbLogWriter(
        str(tmp_path / "phase"),
        "project",
        metric_namespace="phases/01-teacher",
        base_path=str(tmp_path),
    )
    writer.add_scalar("Loss/value", 1.25, global_step=7)
    writer.store_config(_EnvConfig(), {"seed": 42})
    writer.save_model(str(tmp_path / "phase" / "model_7.pt"), 7)
    writer.stop()

    assert fake_wandb.define_metric.call_args_list == [
        call("phases/01-teacher/iteration"),
        call("phases/01-teacher/*", step_metric="phases/01-teacher/iteration"),
    ]
    fake_wandb.log.assert_called_once_with({
        "phases/01-teacher/Loss/value": 1.25,
        "phases/01-teacher/iteration": 7,
    })
    assert (
        call({"phases/01-teacher/train_cfg": {"seed": 42}}, allow_val_change=True)
        in fake_wandb.config.update.call_args_list
    )
    fake_wandb.save.assert_called_once_with(str(tmp_path / "phase" / "model_7.pt"), base_path=str(tmp_path))
    fake_wandb.finish.assert_not_called()
