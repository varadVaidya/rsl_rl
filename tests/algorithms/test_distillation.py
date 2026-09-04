# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Distillation algorithm."""

from __future__ import annotations

import torch
import warnings
from tensordict import TensorDict
from unittest.mock import Mock

import pytest

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 12
OBS_DIM = 8
NUM_ACTIONS = 4


class LatentDistillation(Distillation):
    """Exercise the generic target/prediction seam with a non-action target."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize hook call counters."""
        super().__init__(*args, **kwargs)
        self.target_calls = 0
        self.prediction_calls = 0

    def teacher_target(self, obs: TensorDict) -> torch.Tensor:
        """Produce a latent-sized teacher target."""
        self.target_calls += 1
        return torch.ones(obs.batch_size[0], 6, device=obs.device)

    def student_prediction(self, obs: TensorDict) -> torch.Tensor:
        """Produce the matching latent-sized student prediction."""
        self.prediction_calls += 1
        prediction = self.student(obs)
        return torch.cat((prediction, prediction[..., :2]), dim=-1)


def _make_distillation_setup(
    gradient_length: int = 3,
    num_learning_epochs: int = 1,
    algorithm_class: type[Distillation] = Distillation,
    obs_normalization: bool = False,
) -> tuple:
    """Build a Distillation instance with small networks."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    student = MLPModel(
        obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32], obs_normalization=obs_normalization
    )
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])

    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    alg = algorithm_class(
        student,
        teacher,
        storage,
        num_learning_epochs=num_learning_epochs,
        gradient_length=gradient_length,
        learning_rate=1e-3,
    )
    return alg, obs, storage


def _fill_distillation_storage(alg: Distillation, obs: TensorDict) -> None:
    """Fill the distillation storage with transitions."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.distillation_target = alg.teacher_target(obs).detach()
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


class TestDistillationLoss:
    """Tests for distillation loss computation."""

    def test_loss_decreases_over_updates(self) -> None:
        """Behavior loss should decrease over repeated update() calls (learning signal works)."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3, num_learning_epochs=2)
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_distillation_storage(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["behavior"])

        # Loss should generally decrease; allow some noise — check first vs last
        assert losses[-1] < losses[0], f"Loss should decrease over updates, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_accumulation_step_count(self) -> None:
        """Optimizer should step floor(num_transitions / gradient_length) times per epoch."""
        gradient_length = 4
        alg, obs, _storage = _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=1)
        alg.train_mode()

        _fill_distillation_storage(alg, obs)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()

        expected_steps = NUM_STEPS // gradient_length
        assert step_count == expected_steps, f"Expected {expected_steps} optimizer steps, got {step_count}"

    def test_update_changes_student_but_not_teacher(self) -> None:
        """Student parameters should change after update, while teacher parameters remain frozen."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3)
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.student.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_distillation_storage(alg, obs)
        alg.update()

        any_student_changed = any(
            not torch.equal(p, student_before[name]) for name, p in alg.student.named_parameters()
        )
        assert any_student_changed, "Student parameters should change after an update"

        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during student update"

    def test_custom_targets_and_predictions_need_not_match_action_shape(self) -> None:
        """Subclasses can distill arbitrary targets while collection still emits actions."""
        alg, obs, storage = _make_distillation_setup(gradient_length=NUM_STEPS, algorithm_class=LatentDistillation)
        alg.train_mode()

        for _ in range(NUM_STEPS):
            actions = alg.act(obs)
            assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
            alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})

        assert storage.distillation_target is not None
        assert storage.distillation_target.shape == (NUM_STEPS, NUM_ENVS, 6)
        alg.update()
        assert alg.target_calls == NUM_STEPS
        assert alg.prediction_calls == NUM_STEPS

    def test_eval_student_does_not_update_normalization_during_collection(self) -> None:
        """Frozen/eval students keep collection from invoking normalization updates."""
        alg, obs, _storage = _make_distillation_setup(obs_normalization=True)
        update_normalization = Mock(wraps=alg.student.update_normalization)
        alg.student.update_normalization = update_normalization
        alg.eval_mode()

        alg.act(obs)
        alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})

        update_normalization.assert_not_called()


class TestGradientBudgetWarning:
    """Tests for the construction-time check on the gradient accumulation budget."""

    def test_warns_when_budget_is_not_divisible(self) -> None:
        """A gradient_length of 5 leaves 2 of the 12 rollout steps in an accumulation that is never backpropagated."""
        with pytest.warns(UserWarning, match="The last 2 of 12 steps"):
            _make_distillation_setup(gradient_length=5)

    @pytest.mark.parametrize(
        ("gradient_length", "num_learning_epochs"),
        [(3, 1), (4, 1), (12, 1), (5, 5), (8, 2)],
    )
    def test_no_warning_when_budget_is_divisible(self, gradient_length: int, num_learning_epochs: int) -> None:
        """No warning is raised when the budget divides evenly by gradient_length.

        The budget is num_learning_epochs * num_transitions_per_env, so 5 epochs of 12 steps fit a gradient_length
        of 5 even though a single rollout does not.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=num_learning_epochs)
        assert not [w for w in caught if "gradient_length" in str(w.message)]
