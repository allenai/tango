from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging(enable_cli_logs=True)

    def teardown_method(self):
        super().teardown_method()
        teardown_logging()

    def test_trainer(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations" / "flax" / "config.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.flax",
            ],
        )
        assert (
            result_dir
            / "train"
            / "work"
            / "checkpoint_state_latest"
            / "checkpoint_0"
            / "checkpoint"
        ).is_file()
