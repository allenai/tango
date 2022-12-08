import json
import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, Type, TypeVar, Union, cast
from urllib.parse import ParseResult

from beaker import Digest, Experiment, ExperimentNotFound

from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Run, Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

from ...common.remote_utils import RemoteDatasetNotFound
from .common import BeakerClient, BeakerStepLock, Constants, get_client
from .step_cache import BeakerStepCache

T = TypeVar("T")
U = TypeVar("U", Run, StepInfo)

logger = logging.getLogger(__name__)


@Workspace.register("beaker")
class BeakerWorkspace(RemoteWorkspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on `Beaker`_.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "beaker".

    :param beaker_workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.
    """

    # TODO: merge mem_cache and step_info_cache size use.
    STEP_INFO_CACHE_SIZE = 512
    MEM_CACHE_SIZE = 512
    Constants = Constants

    def __init__(self, workspace: str, max_workers: Optional[int] = None, **kwargs):
        client = get_client(workspace, **kwargs)
        # TODO: for the time being
        cache = BeakerStepCache(workspace, beaker=client.beaker)
        locks: Dict[Step, BeakerStepLock] = {}
        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__(client, cache, "beaker_workspace", locks)
        self.max_workers = max_workers
        self._disk_cache_dir = tango_cache_dir() / "beaker_cache" / "_objects"
        self._mem_cache: "OrderedDict[Digest, Union[StepInfo, Run]]" = OrderedDict()

    @property
    def beaker(self):
        # TODO: for the time being.
        return self.client.beaker

    @property
    def url(self) -> str:
        return f"beaker://{self.beaker.workspace.get().full_name}"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        workspace: str
        if parsed_url.netloc and parsed_url.path:
            # e.g. "beaker://ai2/my-workspace"
            workspace = parsed_url.netloc + parsed_url.path
        elif parsed_url.netloc:
            # e.g. "beaker://my-workspace"
            workspace = parsed_url.netloc
        else:
            raise ValueError(f"Bad URL for Beaker workspace '{parsed_url}'")
        return cls(workspace)

    @property
    def current_beaker_experiment(self) -> Optional[Experiment]:
        """
        When the workspace is being used within a Beaker experiment that was submitted
        by the Beaker executor, this will return the `Experiment` object.
        """
        experiment_name = os.environ.get("BEAKER_EXPERIMENT_NAME")
        if experiment_name is not None:
            try:
                return self.beaker.experiment.get(experiment_name)
            except ExperimentNotFound:
                return None
        else:
            return None

    def _remote_lock(self, step: Step) -> BeakerStepLock:
        self.client = cast(BeakerClient, self.client)
        return BeakerStepLock(
            self.client, step, current_beaker_experiment=self.current_beaker_experiment
        )

    def _dataset_url(self, workspace_url: str, dataset_name: str) -> str:
        return self.client.dataset_url(workspace_url, dataset_name)

    # TODO: make generic.
    def _get_object_from_cache(self, digest: Digest, o_type: Type[U]) -> Optional[U]:
        cache_path = self._disk_cache_dir / make_safe_filename(str(digest))
        if digest in self._mem_cache:
            cached = self._mem_cache.pop(digest)
            # Move to end.
            self._mem_cache[digest] = cached
            return cached if isinstance(cached, o_type) else None
        elif cache_path.is_file():
            try:
                with cache_path.open("r+t") as f:
                    json_dict = json.load(f)
                    cached = o_type.from_json_dict(json_dict)
            except Exception as exc:
                logger.warning("Error while loading object from workspace cache: %s", str(exc))
                try:
                    os.remove(cache_path)
                except FileNotFoundError:
                    pass
                return None
            # Add to in-memory cache.
            self._mem_cache[digest] = cached
            while len(self._mem_cache) > self.MEM_CACHE_SIZE:
                self._mem_cache.popitem(last=False)
            return cached  # type: ignore
        else:
            return None

    def _add_object_to_cache(self, digest: Digest, o: U):
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._disk_cache_dir / make_safe_filename(str(digest))
        self._mem_cache[digest] = o
        with cache_path.open("w+t") as f:
            json.dump(o.to_json_dict(), f)
        while len(self._mem_cache) > self.MEM_CACHE_SIZE:
            self._mem_cache.popitem(last=False)

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        try:
            dataset = self.client.get(self.Constants.step_dataset_name(step_or_unique_id))
            file_info = self.client.file_info(dataset, self.Constants.STEP_INFO_FNAME)
            step_info: StepInfo
            cached = self._get_object_from_cache(file_info.digest, StepInfo)
            if cached is not None:
                step_info = cached
            else:
                step_info_bytes = self.client.get_file(dataset, file_info)
                step_info = StepInfo.from_json_dict(json.loads(step_info_bytes))
                self._add_object_to_cache(file_info.digest, step_info)
            return step_info
        except (RemoteDatasetNotFound, FileNotFoundError):
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info
