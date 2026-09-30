# Copyright (c) 2026 Holiday Robotics
# SPDX-License-Identifier: MIT

"""FlashSAC network building blocks.

Adapted from Holiday-Robot/FlashSAC at commit
87edc9061150ae9e962dd84e6544e27a1554b3ab.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class UnitLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def normalize_parameters(self) -> None:
        self.linear.weight.copy_(F.normalize(self.linear.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        norm = math.sqrt(scale.shape[-1]) * torch.rsqrt(
            torch.sum(scale.square() + bias.square(), dim=-1, keepdim=True) + 1e-8
        )
        self.weight.data.copy_(scale * norm)
        self.bias.data.copy_(bias * norm)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        norm = math.sqrt(scale.shape[-1]) * torch.rsqrt(
            torch.sum(scale.square(), dim=-1, keepdim=True) + 1e-8
        )
        self.weight.data.copy_(scale * norm)


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.linear1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.linear2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.linear1(x), training=training))
        x = F.relu(self.norm2(self.linear2(x), training=training))
        return x + residual


class NormalTanhPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.mean = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

    def get_mean_and_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = F.linear(x, self.mean.linear.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std.linear.weight, self.std_bias)
        log_std = -10.0 + 6.0 * (1.0 + torch.tanh(raw_log_std))
        return mean, torch.exp(log_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.get_mean_and_std(x)
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        log_prob = (
            distribution.log_prob(raw_action) - _safe_tanh_log_det_jacobian(raw_action)
        ).sum(dim=-1)
        return torch.tanh(raw_action), log_prob


class FlashSACActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128, num_blocks: int = 2) -> None:
        super().__init__()
        self.input_norm = UnitBatchNorm(input_dim)
        self.input = UnitLinear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(FlashSACBlock(hidden_dim) for _ in range(num_blocks))
        self.output_norm = UnitRMSNorm(hidden_dim)
        self.policy = NormalTanhPolicy(hidden_dim, action_dim)

    def _features(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.input(self.input_norm(observations, training=training))
        for block in self.blocks:
            x = block(x, training=training)
        return self.output_norm(x)

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.get_mean_and_std(self._features(observations, training))

    def forward(
        self, observations: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(self._features(observations, training))


class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for weight in self.weight:
            nn.init.orthogonal_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))
        self.momentum = momentum

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = x.float()
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                batch_size = x.shape[1]
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                if batch_size > 1:
                    self.running_var.lerp_(
                        (var.squeeze(1) * batch_size / (batch_size - 1)).float(),
                        self.momentum,
                    )
        else:
            mean = self.running_mean.unsqueeze(1)
            var = self.running_var.unsqueeze(1)
        x = (x - mean) * torch.rsqrt(var + 1e-5)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        norm = math.sqrt(scale.shape[-1]) * torch.rsqrt(
            torch.sum(scale.square() + bias.square(), dim=-1, keepdim=True) + 1e-8
        )
        self.weight.data.copy_(scale * norm)
        self.bias.data.copy_(bias * norm)


class EnsembleRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + 1e-6)
        return x / rms * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        norm = math.sqrt(scale.shape[-1]) * torch.rsqrt(
            torch.sum(scale.square(), dim=-1, keepdim=True) + 1e-8
        )
        self.weight.data.copy_(scale * norm)


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.linear1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.linear2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.linear1(x), training=training))
        x = F.relu(self.norm2(self.linear2(x), training=training))
        return x + residual


class FlashSACDoubleCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_bins: int = 101,
        min_value: float = -5.0,
        max_value: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_qs = 2
        total_dim = input_dim + action_dim
        self.input_norm = EnsembleUnitBatchNorm(self.num_qs, total_dim)
        self.input = EnsembleUnitLinear(self.num_qs, total_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            EnsembleFlashSACBlock(self.num_qs, hidden_dim) for _ in range(num_blocks)
        )
        self.output_norm = EnsembleRMSNorm(self.num_qs, hidden_dim)
        self.output = EnsembleUnitLinear(self.num_qs, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(self.num_qs, num_bins))
        self.register_buffer(
            "bin_values", torch.linspace(min_value, max_value, num_bins).reshape(1, 1, -1)
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((observations, actions), dim=-1).unsqueeze(0).expand(self.num_qs, -1, -1)
        x = self.input(self.input_norm(x, training=training))
        for block in self.blocks:
            x = block(x, training=training)
        logits = self.output(self.output_norm(x)) + self.bias.unsqueeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        values = torch.sum(log_probs.exp() * self.bin_values, dim=-1)
        return values, log_probs


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor([math.log(initial_value)]))

    def forward(self) -> torch.Tensor:
        return self.log_temperature.exp()


@torch.no_grad()
def normalize_parameters(module: nn.Module) -> None:
    for child in module.modules():
        fn = getattr(child, "normalize_parameters", None)
        if fn is not None:
            fn()


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, tau: float) -> None:
    torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), tau)
