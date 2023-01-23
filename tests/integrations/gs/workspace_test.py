import os

from tango.common.testing import TangoTestCase
from tango.common.testing.steps import FloatStep
from tango.integrations.gs.common import empty_bucket
from tango.integrations.gs.workspace import GSWorkspace
from tango.step_info import StepState
from tango.workspace import Workspace

GS_BUCKET_NAME = os.environ.get("GS_BUCKET_NAME", "allennlp-tango-bucket")


class TestGSWorkspace(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        empty_bucket(GS_BUCKET_NAME)

    def teardown_method(self):
        super().teardown_method()

    def test_from_url(self):
        workspace = Workspace.from_url(f"gs://{GS_BUCKET_NAME}")
        assert isinstance(workspace, GSWorkspace)

    def test_from_params(self):
        workspace = Workspace.from_params({"type": "gs", "workspace": GS_BUCKET_NAME})
        assert isinstance(workspace, GSWorkspace)

    def test_direct_usage(self):
        workspace = GSWorkspace(GS_BUCKET_NAME)

        step = FloatStep(step_name="float", result=1.0)
        run = workspace.register_run([step])
        assert run.name in workspace.registered_runs()

        assert workspace.step_info(step).state == StepState.INCOMPLETE
        workspace.step_starting(step)
        assert workspace.step_info(step).state == StepState.RUNNING
        workspace.step_finished(step, 1.0)
        assert workspace.step_info(step).state == StepState.COMPLETED
        assert workspace.step_result_for_run(run.name, "float") == 1.0
