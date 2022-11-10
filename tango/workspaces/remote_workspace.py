from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

from tango.common.file_lock import FileLock
from tango.step import Step
from tango.step_cache import StepCache
from tango.workspace import Workspace


class RemoteWorkspace(Workspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on some remote location.

    .. tip::
        All remote workspaces inherit from this.
    """

    def __init__(
        self,
        cache: StepCache,
        steps_dir: Path,
        locks: Dict[Step, FileLock],
        step_info_cache: OrderedDict,
    ):
        self.cache = cache
        self.steps_dir = steps_dir
        self.locks = locks
        self._step_info_cache = step_info_cache

    def _get_unique_id(self, step_or_unique_id: Union[Step, str]) -> str:
        if isinstance(step_or_unique_id, Step):
            unique_id = step_or_unique_id.unique_id
        else:
            unique_id = step_or_unique_id
        return unique_id

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        unique_id = self._get_unique_id(step_or_unique_id)
        path = self.steps_dir / unique_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, step: Step) -> Path:
        path = self.step_dir(step) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def step_cache(self) -> StepCache:
        return self.cache
