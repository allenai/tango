from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, Any, Set, Union, List

from tango.common.exceptions import ConfigurationError
from tango.common.params import Params


@dataclass
class StepStub:
    """
    Stub for a :class:`~tango.step.Step`.
    """

    name: str
    """
    The name of the step.
    """

    config: Dict[str, Any]
    """
    The configuration for the step.
    """

    dependencies: Set[str]
    """
    The other steps that this step directly depends on.
    """


class StepGraph(Sequence):
    """
    Represents an experiment as a directed graph.

    It can be treated as either a :class:`~collections.abc.Mapping` of step names (``str``)
    to :class:`StepStub`, or simply a :class:`~collections.abc.Sequence` of :class:`StepStub`.

    When treated as a sequence, it can be assumed that no step in the sequence depends on a step
    before it.
    """

    def __init__(self, steps: Union[Dict[str, Any], Params]) -> None:
        if isinstance(steps, Params):
            steps = steps.as_dict()
        remaining_steps_to_sort: Dict[str, StepStub] = {}
        for step_name, step_config in steps.items():
            dependencies = self._parse_direct_step_dependencies(step_config)
            remaining_steps_to_sort[step_name] = StepStub(
                name=step_name, config=step_config, dependencies=dependencies
            )

        # These are the steps in the order that they should run.
        self._ordered_steps: Dict[str, StepStub] = OrderedDict()
        for _ in range(len(remaining_steps_to_sort)):
            # Go through sorted to ensure this is deterministic.
            for step_name in sorted(remaining_steps_to_sort.keys()):
                step_stub = remaining_steps_to_sort[step_name]
                for ref in step_stub.dependencies:
                    if ref not in self._ordered_steps:
                        # Step depends an other later step, so it needs to wait.
                        break
                else:
                    self._ordered_steps[step_name] = step_stub
                    remaining_steps_to_sort.pop(step_name)
                    break

        # Validate the graph.
        if remaining_steps_to_sort:
            err_msgs: List[str] = []
            for step_name, step_stub in remaining_steps_to_sort.items():
                for ref in step_stub.dependencies:
                    if ref not in self._ordered_steps:
                        err_msgs.append(f"Can't resolve dependency {ref} for {step_name}")

            raise ConfigurationError("Invalid step graph:\n- " + "\n- ".join(err_msgs))

    def __getitem__(self, key: Union[str, int]) -> StepStub:  # type: ignore[override]
        """
        Get the stub corresponding to ``key``.
        """
        if isinstance(key, str):
            return self._ordered_steps[key]
        else:
            return self._ordered_steps[list(self._ordered_steps.keys())[key]]

    def __len__(self) -> int:
        """
        The number of steps in the experiment.
        """
        return len(self._ordered_steps)

    @staticmethod
    def _parse_direct_step_dependencies(o: Any) -> Set[str]:
        dependencies: Set[str] = set()
        if isinstance(o, (list, tuple, set)):
            for item in o:
                dependencies = dependencies | StepGraph._parse_direct_step_dependencies(item)
        elif isinstance(o, dict):
            if set(o.keys()) == {"type", "ref"}:
                dependencies.add(o["ref"])
            else:
                for value in o.values():
                    dependencies = dependencies | StepGraph._parse_direct_step_dependencies(value)
        elif o is not None and not isinstance(o, (bool, str, int, float)):
            raise ValueError(o)
        return dependencies
