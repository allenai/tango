import atexit
import json
import logging
import tempfile
import time
import urllib
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from beaker import Beaker
from beaker import Dataset as BeakerDataset
from beaker import (
    DatasetConflict,
    DatasetNotFound,
    DatasetWriteError,
    Experiment,
    ExperimentNotFound,
    FileInfo,
)

from tango.common.aliases import PathOrStr
from tango.common.remote_utils import (
    RemoteClient,
    RemoteConstants,
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
    RemoteDatasetWriteError,
    RemoteStepLock,
)
from tango.step import Step
from tango.step_info import StepInfo
from tango.version import VERSION

# from tango.common.util import utc_now_datetime

logger = logging.getLogger(__name__)


class BeakerClient(RemoteClient):
    """
    A client for interacting with beaker.
    """

    def __init__(self, beaker_workspace: Optional[str] = None, **kwargs):
        user_agent = f"tango v{VERSION}"
        if beaker_workspace is not None:
            self.beaker = Beaker.from_env(
                default_workspace=beaker_workspace,
                session=True,
                user_agent=user_agent,
                **kwargs,
            )
        else:
            self.beaker = Beaker.from_env(session=True, user_agent=user_agent, **kwargs)

    @property
    def full_name(self):
        return self.beaker.workspace.get().full_name

    def url(self, dataset: Optional[str] = None):
        return self.beaker.dataset.url(dataset)

    def dataset_url(self, workspace_url: str, dataset_name: str) -> str:
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

    def get(self, dataset: Union[str, BeakerDataset]) -> BeakerDataset:
        try:
            return self.beaker.dataset.get(dataset)
        except DatasetConflict:
            # We do this so that remote_workspace gets errors of the type RemoteDatasetNotFound.
            raise RemoteDatasetNotFound()

    def create(self, dataset: str, commit: bool = False):
        try:
            self.beaker.dataset.create(dataset, commit=commit)
        except DatasetConflict:
            raise RemoteDatasetConflict()

    def delete(self, dataset: BeakerDataset):
        self.beaker.dataset.delete(dataset)

    def sync(self, dataset: Union[str, BeakerDataset], objects_dir: Path):
        try:
            self.beaker.dataset.sync(dataset, objects_dir, quiet=True)
        except DatasetWriteError:
            raise RemoteDatasetWriteError()

    def commit(self, dataset: Union[str, BeakerDataset]):
        self.beaker.dataset.commit(dataset)

    def upload(self, dataset: BeakerDataset, source: bytes, target: PathOrStr) -> None:
        self.beaker.dataset.upload(dataset, source, target)

    def get_file(self, dataset: BeakerDataset, file_path: Union[str, FileInfo]):
        try:
            self.beaker.dataset.get_file(dataset, file_path, quiet=True)
        except DatasetNotFound:
            raise RemoteDatasetNotFound()

    def file_info(self, dataset: BeakerDataset, file_path: str) -> FileInfo:
        try:
            return self.beaker.dataset.file_info(dataset, file_path)
        except DatasetNotFound:
            raise RemoteDatasetNotFound()

    def fetch(self, dataset: BeakerDataset, target_dir: PathOrStr):
        try:
            self.beaker.dataset.fetch(dataset, target_dir, quiet=True)
        except DatasetNotFound:
            raise RemoteDatasetNotFound()

    def datasets(
        self, match: str, uncommitted: bool = True, results: bool = False
    ) -> List[BeakerDataset]:
        return self.beaker.workspace.iter_datasets(
            match=match, uncommitted=uncommitted, results=results
        )


class Constants(RemoteConstants):
    ENTRYPOINT_DATASET_PREFIX = "tango-entrypoint-"
    BEAKER_TOKEN_SECRET_NAME: str = "BEAKER_TOKEN"
    ENTRYPOINT_DIR: str = "/tango/entrypoint"
    ENTRYPOINT_FILENAME: str = "entrypoint.sh"


def get_client(beaker_workspace: Optional[str] = None, **kwargs) -> BeakerClient:
    return BeakerClient(beaker_workspace, **kwargs)


class BeakerStepLock(RemoteStepLock):
    def __init__(
        self,
        client: BeakerClient,
        step: Union[str, StepInfo, Step],
        current_beaker_experiment: Optional[Experiment] = None,
    ):
        super().__init__(client, step)
        self._current_beaker_experiment = current_beaker_experiment

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "beaker_experiment": None
            if not self._current_beaker_experiment
            else self._current_beaker_experiment.id
        }

    def _last_metadata(self) -> Optional[Dict[str, Any]]:
        try:
            metadata_bytes = self._client.get_file(self._lock_dataset_name, self.METADATA_FNAME)
            metadata = json.loads(metadata_bytes)
            return metadata
        except (RemoteDatasetNotFound, FileNotFoundError):
            return None

    def _acquiring_job_is_done(self) -> bool:
        last_metadata = self._last_metadata()
        if last_metadata is None:
            return False

        last_experiment_id = last_metadata.get("beaker_experiment")
        if last_experiment_id is None:
            return False

        try:
            last_experiment = self._client.beaker.experiment.get(last_experiment_id)  # type: ignore
            if (
                self._current_beaker_experiment is not None
                and self._current_beaker_experiment.id == last_experiment_id
            ):
                # This means a previous job for this experiment was preempted and
                # it didn't clean up after itself.
                return True
            else:
                job = self._client.beaker.experiment.latest_job(last_experiment)  # type: ignore
                return False if job is None else job.is_done
        except ExperimentNotFound:
            # Experiment must have been deleted.
            return True
        except ValueError:
            return False

    def acquire(self, timeout=None, poll_interval: float = 2.0, log_interval: float = 30.0) -> None:
        if self._lock_dataset is not None:
            return
        start = time.monotonic()
        last_logged = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_dataset = self._client.create(self._lock_dataset_name, commit=False)

                atexit.register(self.release)

                # Write metadata.
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tmp_dir = Path(tmp_dir_name)
                    metadata_path = tmp_dir / self.METADATA_FNAME
                    with open(metadata_path, "w") as f:
                        json.dump(self.metadata, f)
                    self._client.sync(self._lock_dataset, metadata_path)
            except RemoteDatasetConflict:
                # Check if existing lock was created from a Beaker experiment.
                # If it was, and the experiment is no-longer running, we can safely
                # delete it.
                if self._acquiring_job_is_done():
                    self._client.delete(self._lock_dataset_name)
                    continue

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
