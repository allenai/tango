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
            self.FIXTURES_ROOT / "integrations/pytorch_lightning/train.jsonnet",
            include_package=["test_fixtures.integrations.pytorch_lightning"],
            overrides="" if with_validation else "{'steps.train.validation_split':null}",
        )

        assert (result_dir / "train" / "data.pt").is_file()
        assert (result_dir / "train" / "work" / "epoch=4-step=39.ckpt").is_file()
