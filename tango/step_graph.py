import logging
from typing import Any, Dict, Iterator, List, Mapping, Set

from tango.common.exceptions import ConfigurationError
from tango.common.params import Params
from tango.step import Step

logger = logging.getLogger(__name__)


class StepGraph(Mapping[str, Step]):
    """
    Represents an experiment as a directed graph.

    It can be treated as a :class:`~collections.abc.Mapping` of step names (``str``)
    to :class:`Step`.
    """

    def __init__(self, params: Dict[str, Params]):
        # TODO: What happens with anonymous steps in here?

        # Determine the order in which to create steps so that all dependent steps are available when we need them.
        # This algorithm for resolving step dependencies is O(n^2). Since we're
        # anticipating the number of steps in a single config to be in the dozens at most (#famouslastwords),
        # we choose simplicity over cleverness.
        dependencies = {
            step_name: self._find_step_dependencies(step_params)
            for step_name, step_params in params.items()
        }

        # Check whether some of those dependencies can never be satisfied.
        unsatisfiable_dependencies = {
            dep
            for step_deps in dependencies.values()
            for dep in step_deps
            if dep not in dependencies.keys()
        }
        if len(unsatisfiable_dependencies) > 0:
            if len(unsatisfiable_dependencies) == 1:
                dep = next(iter(unsatisfiable_dependencies))
                raise ConfigurationError(
                    f"Specified dependency '{dep}' can't be found in the config."
                )
            else:
                raise ConfigurationError(
                    f"Some dependencies can't be found in the config: {', '.join(unsatisfiable_dependencies)}"
                )

        done: Set[str] = set()
        todo = list(params.keys())
        ordered_steps = list()
        while len(todo) > 0:
            new_todo = []
            for step_name in todo:
                if len(dependencies[step_name] & done) == len(dependencies[step_name]):
                    done.add(step_name)
                    ordered_steps.append(step_name)
                else:
                    new_todo.append(step_name)
            if len(todo) == len(new_todo):
                raise ConfigurationError(
                    "Could not make progress parsing the steps. "
                    "You probably have a circular reference between the steps."
                )
            todo = new_todo
        del dependencies
        del done
        del todo

        # Parse the steps
        self.parsed_steps: Dict[str, Step] = {}
        for step_name in ordered_steps:
            step_params = params.pop(step_name)
            if step_name in self.parsed_steps:
                raise ConfigurationError(f"Duplicate step name {step_name}")

            step_params = self._replace_step_dependencies(step_params, self.parsed_steps)
            self.parsed_steps[step_name] = Step.from_params(step_params, step_name=step_name)

        # Sanity-check the graph
        for step in self.parsed_steps.values():
            if step.cache_results:
                nondeterministic_dependencies = [
                    s for s in step.recursive_dependencies if not s.DETERMINISTIC
                ]
                if len(nondeterministic_dependencies) > 0:
                    nd_step = nondeterministic_dependencies[0]
                    logger.warning(
                        f"Task {step.name} is set to cache results, but depends on non-deterministic "
                        f"step {nd_step.name}. This will produce confusing results."
                    )

    @staticmethod
    def _dict_is_ref(d: dict) -> bool:
        keys = set(d.keys())
        if keys == {"ref"}:
            return True
        if keys == {"type", "ref"} and d["type"] == "ref":
            return True
        return False

    @classmethod
    def _find_step_dependencies(cls, o: Any) -> Set[str]:
        dependencies: Set[str] = set()
        if isinstance(o, (list, tuple, set)):
            for item in o:
                dependencies = dependencies | cls._find_step_dependencies(item)
        elif isinstance(o, dict):
            if cls._dict_is_ref(o):
                dependencies.add(o["ref"])
            else:
                for value in o.values():
                    dependencies = dependencies | cls._find_step_dependencies(value)
        elif o is not None and not isinstance(o, (bool, str, int, float)):
            raise ValueError(o)
        return dependencies

    @classmethod
    def _replace_step_dependencies(cls, o: Any, existing_steps: Mapping[str, Step]) -> Any:
        if isinstance(o, (list, tuple, set)):
            return o.__class__(cls._replace_step_dependencies(i, existing_steps) for i in o)
        elif isinstance(o, dict):
            if cls._dict_is_ref(o):
                return existing_steps[o["ref"]]
            else:
                return {
                    key: cls._replace_step_dependencies(value, existing_steps)
                    for key, value in o.items()
                }
        elif o is not None and not isinstance(o, (bool, str, int, float)):
            raise ValueError(o)
        return o

    def __getitem__(self, name: str) -> Step:
        """
        Get the step with the given name.
        """
        return self.parsed_steps[name]

    def __len__(self) -> int:
        """
        The number of steps in the experiment.
        """
        return len(self.parsed_steps)

    def __iter__(self) -> Iterator[str]:
        """
        The names of the steps in the experiment.
        """
        return iter(self.parsed_steps)

    def ordered_steps(self) -> List[Step]:
        """
        Returns the steps in this step graph in an order that can be executed one at a time.

        This does not take into account which steps may be cached. It simply returns an executable
        order of steps.
        """
        result: List[Step] = []
        steps_run: Set[Step] = set()
        steps_not_run = list(self.parsed_steps.values())
        while len(steps_not_run) > 0:
            step = steps_not_run.pop(0)
            if len(step.dependencies & steps_run) == len(step.dependencies):
                steps_run.add(step)
                result.append(step)
            else:
                steps_not_run.append(step)
        return result
