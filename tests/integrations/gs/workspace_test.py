import os

import pytest

from tango.common.testing import TangoTestCase
from tango.common.testing.steps import FloatStep
from tango.integrations.gs.common import empty_bucket_folder, empty_datastore
from tango.integrations.gs.workspace import GSWorkspace
from tango.step_info import StepState
from tango.workspace import Workspace

GS_BUCKET_NAME = os.environ.get("GS_BUCKET_NAME", "allennlp-tango-bucket")
GS_SUBFOLDER = f"{GS_BUCKET_NAME}/my-workspaces/workspace1"


class TestGSWorkspace(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        empty_bucket_folder(GS_BUCKET_NAME)
        empty_bucket_folder(GS_SUBFOLDER)
        empty_datastore(GS_BUCKET_NAME)
        empty_datastore(GS_SUBFOLDER)

    def teardown_method(self):
        super().teardown_method()

    @pytest.mark.parametrize("gs_path", [GS_BUCKET_NAME, GS_SUBFOLDER])
    def test_from_url(self, gs_path: str):
        workspace = Workspace.from_url(f"gs://{gs_path}")
        assert isinstance(workspace, GSWorkspace)

    @pytest.mark.parametrize("gs_path", [GS_BUCKET_NAME, GS_SUBFOLDER])
    def test_from_params(self, gs_path: str):
        workspace = Workspace.from_params({"type": "gs", "workspace": gs_path})
        assert isinstance(workspace, GSWorkspace)

    @pytest.mark.parametrize("gs_path", [GS_BUCKET_NAME, GS_SUBFOLDER])
    def test_direct_usage(self, gs_path: str):
        workspace = GSWorkspace(gs_path)

        step = FloatStep(step_name="float", result=1.0)
        run = workspace.register_run([step])
        assert run.name in workspace.registered_runs()

        assert workspace.step_info(step).state == StepState.INCOMPLETE
        workspace.step_starting(step)
        assert workspace.step_info(step).state == StepState.RUNNING
        workspace.step_finished(step, 1.0)
        assert workspace.step_info(step).state == StepState.COMPLETED
        assert workspace.step_result_for_run(run.name, "float") == 1.0

    def test_remove_step(self):
        workspace = GSWorkspace(GS_BUCKET_NAME)
        step = FloatStep(step_name="float", result=1.0)
        step_info = workspace.step_info(step)

        workspace.step_starting(step)
        workspace.step_finished(step, 1.0)
        bucket_artifact = workspace.Constants.step_artifact_name(step_info)
        ds_entity = workspace._ds.get(key=workspace._ds.key("stepinfo", step_info.unique_id))
        cache = workspace.step_cache

        assert workspace.client.artifacts(prefix=bucket_artifact) is not None
        assert ds_entity is not None
        assert step in cache

        workspace.remove_step(step.unique_id)
        cache = workspace.step_cache

        ds_entity = workspace._ds.get(key=workspace._ds.key("stepinfo", step_info.unique_id))

        with pytest.raises(Exception) as excinfo:
            workspace.client.artifacts(prefix=bucket_artifact)

        assert "KeyError" in str(excinfo)
        assert ds_entity is None
        assert step not in cache
