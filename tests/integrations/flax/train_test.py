import json
import pytest
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        initialize_logging(enable_cli_logs=True)

    def teardown_method(self):
        super().teardown_method()
        teardown_logging()

    def test_basic_train(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations" / "flax" / "train.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.flax",
            ],
            overrides=""
        )

