import atexit
import json
import logging
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, Union

from beaker import (
    Beaker,
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    Experiment,
    ExperimentNotFound,
)

from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class Constants:
    RUN_DATASET_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_DATASET_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"
    ENTRYPOINT_DATASET_PREFIX = "tango-entrypoint-"
    STEP_GRAPH_DATASET_PREFIX = "tango-step-graph-"
    STEP_EXPERIMENT_PREFIX = "tango-step-"


def step_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


def step_lock_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{step_dataset_name(step)}-lock"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"


def dataset_url(workspace_url: str, dataset_name: str) -> str:
    return (
        workspace_url
        + "/datasets?"
        + urllib.parse.urlencode(
            {
                "text": dataset_name,
                "committed": "false",
            }
        )
    )


class BeakerStepLock:
    METADATA_FNAME = "metadata.json"

    def __init__(
        self,
        beaker: Beaker,
        step: Union[str, StepInfo, Step],
        current_beaker_experiment: Optional[Experiment] = None,
    ):
        self._beaker = beaker
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._lock_dataset_name = step_lock_dataset_name(step)
        self._lock_dataset: Optional[Dataset] = None
        self._current_beaker_experiment = current_beaker_experiment
        self.lock_dataset_url = dataset_url(beaker.workspace.url(), self._lock_dataset_name)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "beaker_experiment": None
            if not self._current_beaker_experiment
            else self._current_beaker_experiment.id
        }

    def acquire(self, timeout=None, poll_interval: float = 2.0, log_interval: float = 30.0) -> None:
        if self._lock_dataset is not None:
            return
        start = time.monotonic()
        last_logged = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_dataset = self._beaker.dataset.create(
                    self._lock_dataset_name, commit=False
                )

                atexit.register(self.release)

                # Write metadata.
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tmp_dir = Path(tmp_dir_name)
                    metadata_path = tmp_dir / self.METADATA_FNAME
                    with open(metadata_path, "w") as f:
                        json.dump(self.metadata, f)
                    self._beaker.dataset.sync(self._lock_dataset, metadata_path, quiet=True)
            except DatasetConflict:
                # Check if existing lock was created from a Beaker experiment.
                # If it was, and the experiment is no-longer running, we can safely
                # delete it.
                try:
                    metadata_bytes = self._beaker.dataset.get_file(
                        self._lock_dataset_name, self.METADATA_FNAME, quiet=True
                    )
                    metadata = json.loads(metadata_bytes)
                    experiment_id = metadata.get("beaker_experiment")
                    if experiment_id is not None:
                        try:
                            experiment = self._beaker.experiment.get(experiment_id)
                            job = self._beaker.experiment.latest_job(experiment)
                            if job is not None and job.is_done:
                                self._beaker.dataset.delete(self._lock_dataset_name)
                                continue
                        except ExperimentNotFound:
                            self._beaker.dataset.delete(self._lock_dataset_name)
                            continue
                except DatasetNotFound:
                    continue
                except Exception:
                    pass

                if last_logged is None or last_logged - start >= log_interval:
                    logger.warning(
                        "Waiting to acquire lock dataset for step '%s':\n\n%s\n\n"
                        "This probably means the step is being run elsewhere, but if you're sure it isn't "
                        "you can just delete the lock dataset.",
                        self._step_id,
                        self.lock_dataset_url,
                    )
                    last_logged = time.monotonic()
                time.sleep(poll_interval)
                continue
            else:
                break
        else:
            raise TimeoutError(
                f"Timeout error occurred while waiting to acquire dataset lock for step '{self._step_id}':\n\n"
                f"{self.lock_dataset_url}\n\n"
                f"This probably means the step is being run elsewhere, but if you're sure it isn't you can "
                f"just delete the lock dataset."
            )

    def release(self):
        if self._lock_dataset is not None:
            try:
                self._beaker.dataset.delete(self._lock_dataset)
            except DatasetNotFound:
                # Dataset must have been manually deleted.
                pass
            self._lock_dataset = None
            atexit.unregister(self.release)

    def __del__(self):
        self.release()
