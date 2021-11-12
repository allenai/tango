import pytest
import torch.distributed as dist

from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    def teardown_method(self):
        super().teardown_method()
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.parametrize("with_validation", [True, False])
    def test_basic_train(self, with_validation):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train.jsonnet",
            include_package=["test_fixtures.integrations.torch"],
            overrides="" if with_validation else "{'steps.train.validation_split':null}",
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

    def test_train_distributed(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train_dist.jsonnet",
            include_package=["test_fixtures.integrations.torch"],
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
