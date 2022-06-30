from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase


class TestTrainStep(TangoTestCase):
    # def setup_method(self):
    #     super().setup_method()
    #     initialize_logging(enable_cli_logs=True)
    #
    # def teardown_method(self):
    #     super().teardown_method()
    #     teardown_logging()
    #
    # def test_mnist_train(self):
    #     result_dir = self.run(
    #         self.FIXTURES_ROOT / "integrations" / "flax" / "mnist.jsonnet",
    #         include_package=[
    #             "test_fixtures.integrations.common",
    #             "test_fixtures.integrations.flax",
    #         ],
    #         overrides="",
    #     )

        # assert (result_dir / "checkpoint" / "1000").is_file()

    def test_transformer_train(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations" / "flax" / "transformer.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.flax",
            ],
            overrides="",
        )