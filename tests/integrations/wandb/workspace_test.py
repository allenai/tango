import json
import os
import pickle
import shutil
import sys
import uuid

import pytest
import wandb

from tango import Step, StepGraph, Workspace
from tango.common import Params, util
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase
from tango.integrations.wandb import WandbWorkspace
from tango.step_info import StepState
from test_fixtures.package.steps import *  # noqa: F403,F401

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "allennlp")
WANDB_PROJECT = "tango-workspace-testing"


class TestWandbWorkspace(TangoTestCase):
    # Need to define the `setup_method()` as fixture so we can use other fixtures within it.
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch):
        super().setup_method()
        # Patch tango_cache_dir()
        monkeypatch.setattr(util, "tango_cache_dir", lambda: self.TEST_DIR)

    @pytest.mark.parametrize(
        "protocol",
        [pytest.param(protocol, id=f"protocol={protocol}") for protocol in range(4)]
        + [
            pytest.param(
                5,
                id="protocol=5",
                marks=pytest.mark.skipif(
                    sys.version_info < (3, 8), reason="Protocol 5 requires Python 3.8 or newer"
                ),
            ),
        ],
    )
    def test_pickle_workspace(self, protocol):
        workspace = WandbWorkspace(project=WANDB_PROJECT, entity=WANDB_ENTITY)
        unpickled_workspace = pickle.loads(pickle.dumps(workspace, protocol=protocol))
        assert unpickled_workspace.wandb_client is not None
        assert unpickled_workspace.project == workspace.project
        assert unpickled_workspace.entity == workspace.entity
        assert unpickled_workspace.steps_dir == workspace.steps_dir

    def test_from_url(self):
        workspace = Workspace.from_url(f"wandb://{WANDB_ENTITY}/{WANDB_PROJECT}")
        assert isinstance(workspace, WandbWorkspace)
        assert workspace.entity == WANDB_ENTITY
        assert workspace.project == WANDB_PROJECT


class TestWandbWorkspaceUsage(TangoTestCase):
    # Need to define the `setup_method()` as fixture so we can use other fixtures within it.
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch):
        super().setup_method()
        self.UNIQUE_ID_SUFFIX = os.environ.get("GITHUB_SHA", "")[:6] + "-" + str(uuid.uuid1())[:6]
        # Patch tango_cache_dir()
        monkeypatch.setattr(util, "tango_cache_dir", lambda: self.TEST_DIR)
        # Patch Step unique IDs and W&B run IDs.
        monkeypatch.setattr(Step, "_UNIQUE_ID_SUFFIX", self.UNIQUE_ID_SUFFIX)
        monkeypatch.setattr(
            WandbWorkspace,
            "_generate_run_suite_id",
            lambda workspace: wandb.util.generate_id() + "-" + self.UNIQUE_ID_SUFFIX,
        )

        self.workspace = WandbWorkspace(project=WANDB_PROJECT, entity=WANDB_ENTITY)

        initialize_logging(enable_cli_logs=True)

    def teardown_method(self):
        super().teardown_method()

        # Delete W&B runs and their artifacts produced by the test.
        for wandb_run in self.workspace.wandb_client.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        ):
            if (
                self.UNIQUE_ID_SUFFIX in wandb_run.id
                or self.UNIQUE_ID_SUFFIX in wandb_run.config.get("_run_suite_id", "")
            ):
                wandb_run.delete(delete_artifacts=True)

        teardown_logging()

    def test_direct_usage(self):
        params = Params.from_file(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet")
        step_graph = StepGraph.from_params(params.pop("steps", keep_as_dict=True))
        tango_run = self.workspace.register_run(step for step in step_graph.values())

        # Test 'registered_run()' and 'registered_runs()' methods.
        assert self.workspace.registered_run(tango_run.name) == tango_run
        assert self.workspace.registered_runs()[tango_run.name] == tango_run

        hello_step = step_graph["hello"]
        hello_world_step = step_graph["hello_world"]

        # Test getting step info.
        step_info = self.workspace.step_info(hello_step)
        assert step_info.unique_id.endswith(self.UNIQUE_ID_SUFFIX)
        assert step_info.step_name == "hello"
        assert step_info.state == StepState.INCOMPLETE

        # Mark the "hello" step as starting.
        self.workspace.step_starting(hello_step)
        assert self.workspace.step_info(hello_step).state == StepState.RUNNING

        # Mark the "hello" step as finished.
        self.workspace.step_finished(hello_step, "hello")
        assert self.workspace.step_info(hello_step).state == StepState.COMPLETED

        # Make sure the result is in the cache, exists locally, and on W&B.
        cache = self.workspace.cache
        assert hello_step in cache
        assert cache.step_dir(hello_step).is_dir()
        assert cache.get_step_result_artifact(hello_step) is not None

        # Now make sure we can fetch the item from the cache, even if it's not in memory
        # or in the cache directory.
        if hello_step.unique_id in cache.weak_cache:
            del cache.weak_cache[hello_step.unique_id]
        if hello_step.unique_id in cache.strong_cache:
            del cache.strong_cache[hello_step.unique_id]
        shutil.rmtree(cache.step_dir(hello_step))
        assert hello_step in cache
        assert cache[hello_step] == "hello"

        # Now start the "hello_world" step and then mark it as failed.
        self.workspace.step_starting(hello_world_step)
        self.workspace.step_failed(hello_world_step, ValueError("oh no!"))
        assert self.workspace.step_info(hello_world_step).state == StepState.FAILED

    @pytest.mark.parametrize(
        "multicore", [pytest.param(True, id="multicore"), pytest.param(False, id="singe-core")]
    )
    @pytest.mark.parametrize(
        "distributed",
        [
            pytest.param(True, id="distributed"),
            pytest.param(False, id="single-device"),
        ],
    )
    def test_with_wandb_train_callback(self, multicore: bool, distributed: bool):
        self.run(
            self.FIXTURES_ROOT
            / "integrations"
            / "torch"
            / ("train.jsonnet" if not distributed else "train_dist.jsonnet"),
            include_package=[
                "test_fixtures.integrations.common",
                "test_fixtures.integrations.torch",
            ],
            overrides=json.dumps({"steps.train.callbacks": [{"type": "wandb::log"}]}),
            workspace_url=f"wandb://{WANDB_ENTITY}/{WANDB_PROJECT}",
            multicore=multicore,
        )
