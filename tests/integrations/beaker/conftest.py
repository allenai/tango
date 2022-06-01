import os
import uuid
from pathlib import Path
from typing import Generator

import pytest
from beaker import Beaker

from tango.common import util
from tango.integrations.beaker.common import Constants
from tango.step import Step


@pytest.fixture(autouse=True)
def patched_cache_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(util, "tango_cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def patched_unique_id_suffix(monkeypatch) -> str:
    UNIQUE_ID_SUFFIX = os.environ.get("GITHUB_SHA", "")[:6] + "-" + str(uuid.uuid1())[:6]
    monkeypatch.setattr(Step, "_UNIQUE_ID_SUFFIX", UNIQUE_ID_SUFFIX)
    return UNIQUE_ID_SUFFIX


@pytest.fixture(autouse=True)
def patched_constants_prefix(monkeypatch) -> str:
    PREFIX = os.environ.get("GITHUB_SHA", "A")[:6] + "-" + str(uuid.uuid1())[:6] + "-"
    monkeypatch.setattr(Constants, "STEP_DATASET_PREFIX", "tango-step-" + PREFIX)
    monkeypatch.setattr(Constants, "RUN_DATASET_PREFIX", "tango-run-" + PREFIX)
    return PREFIX


@pytest.fixture
def beaker_workspace(
    patched_unique_id_suffix: str, patched_constants_prefix: str
) -> Generator[str, None, None]:
    workspace_name = "ai2/tango-beaker-testing"
    beaker = Beaker.from_env(default_workspace=workspace_name)
    yield workspace_name
    # Remove datasets.
    for dataset in beaker.workspace.datasets(match=patched_unique_id_suffix):
        beaker.dataset.delete(dataset)
    for dataset in beaker.workspace.datasets(match=patched_constants_prefix):
        beaker.dataset.delete(dataset)
