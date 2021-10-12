from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    def test_basic_train_loop(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train.jsonnet",
            include_package=["test_fixtures.integrations.torch"],
        )
        assert (result_dir / "train" / "data.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker0.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker0_best.pt").is_file()

    def test_train_distributed(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/train_dist.jsonnet",
            include_package=["test_fixtures.integrations.torch"],
        )
        assert (result_dir / "train" / "data.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker0.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker0_best.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker1.pt").is_file()
        assert (result_dir / "train" / "work" / "state_worker1_best.pt").is_file()
