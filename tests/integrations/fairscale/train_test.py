from typing import Any, Dict

import pytest
import torch

from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestFairScaleTrain(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging(log_level="info")

    def teardown_method(self):
        teardown_logging()

    @pytest.mark.parametrize(
        "fsdp",
        (
            pytest.param(
                True,
                id="fsdp=True",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        torch.cuda.device_count() < 2, reason="Requires CUDA devices"
                    ),
                ],
            ),
            pytest.param(False, id="fsdp=False"),
        ),
    )
    @pytest.mark.parametrize(
        "activation_checkpoint",
        (
            pytest.param(True, id="checkpointing=True"),
            pytest.param(False, id="checkpointing=False"),
        ),
    )
    @pytest.mark.parametrize(
        "amp",
        (
            pytest.param(
                True,
                id="amp=True",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        torch.cuda.device_count() < 2, reason="Requires CUDA devices"
                    ),
                ],
            ),
            pytest.param(False, id="amp=False"),
        ),
    )
    def test_train_tiny_gpt2(self, fsdp: bool, activation_checkpoint: bool, amp: bool):
        overrides: Dict[str, Any] = {
            "steps.trained_model.model.activation_checkpointing": activation_checkpoint,
        }
        training_engine: Dict[str, Any] = {
            "amp": amp,
            "optimizer": {
                "type": "torch::AdamW",
                "lr": 0.005,
                "betas": [0.9, 0.95],
                "eps": 1e-6,
            },
        }
        if fsdp:
            training_engine["type"] = "fairscale"
            fsdp_config = {"reshard_after_forward": True, "mixed_precision": amp}
            training_engine["fsdp_config"] = fsdp_config
            overrides["steps.trained_model.model.fsdp_config"] = fsdp_config
        else:
            training_engine["type"] = "torch"
            overrides["steps.trained_model.model.fsdp_config"] = None
        overrides["steps.trained_model.training_engine"] = training_engine
        run_dir = self.run(
            self.FIXTURES_ROOT / "integrations" / "fairscale" / "config.jsonnet",
            include_package=["test_fixtures.integrations.fairscale.components"],
            overrides=overrides,
        )
        assert (run_dir / "trained_model").is_dir()
