from tango.common.testing import TangoTestCase


class TestEvalStep(TangoTestCase):
    def test_basic_eval(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations/torch/eval.jsonnet",
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
        )
        assert (result_dir / "train" / "data.json").is_file()
