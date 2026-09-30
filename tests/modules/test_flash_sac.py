# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from rsl_rl.modules.flash_sac import EnsembleUnitBatchNorm


def test_ensemble_batch_norm_keeps_large_half_precision_inputs_finite() -> None:
    """Compute normalization statistics in float32 under AMP."""
    layer = EnsembleUnitBatchNorm(num_ensemble=2, input_dim=4)
    inputs = torch.tensor(
        [[[-1000.0] * 4, [1000.0] * 4]] * 2,
        dtype=torch.float16,
    )

    outputs = layer(inputs, training=True)

    assert torch.isfinite(outputs).all()
    assert torch.isfinite(layer.running_var).all()
