import json

import pytest
import torch.distributed as dist

from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging(enable_cli_logs=True)

    def teardown_method(self):
        super().teardown_method()
        if dist.is_initialized():
            dist.destroy_process_group()
        teardown_logging()

    @pytest.mark.parametrize("with_validation", [True, False])
    def test_basic_train(self, with_validation: bool):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
            overrides=""
            if with_validation
            else json.dumps(
                {"steps.train.validation_split": None, "steps.train.validate_every": None}
            ),
        )
        assert (result_dir / "train" / "data.pt").is_file()
        assert (result_dir / "train" / "work" / "weights.pt").is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_latest" / "worker0_model.pt"
        ).is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_best" / "worker0_optimizer.pt"
        ).is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_best" / "worker0_trainer.pt"
        ).is_file()

    def test_basic_train_with_epochs(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
            overrides=json.dumps(
                {
                    "steps.train.train_steps": None,
                    "steps.train.train_epochs": 2,
                    "steps.train.validate_every": None,
                }
            ),
        )
        assert (result_dir / "train" / "data.pt").is_file()

    def test_basic_train_with_streaming_data(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
        )
        assert (result_dir / "train" / "data.pt").is_file()

    def test_train_distributed(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train_dist.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
        )
        assert (result_dir / "train" / "data.pt").is_file()
        assert (result_dir / "train" / "work" / "weights.pt").is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_latest" / "worker0_model.pt"
        ).is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_best" / "worker0_model.pt"
        ).is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_latest" / "worker1_model.pt"
        ).is_file()
        assert (
            result_dir / "train" / "work" / "checkpoint_state_best" / "worker1_model.pt"
        ).is_file()
