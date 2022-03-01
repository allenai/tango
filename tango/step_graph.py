import logging
from typing import Any, Dict, Iterator, List, Mapping, Set, Type

from tango.common import PathOrStr
from tango.common.exceptions import ConfigurationError
from tango.common.params import Params
from tango.common.util import import_extra_module
from tango.step import Step

logger = logging.getLogger(__name__)


class StepGraph(Mapping[str, Step]):
    """
    Represents an experiment as a directed graph.

    It can be treated as a :class:`~collections.abc.Mapping` of step names (``str``)
    to :class:`Step`.
    """

    def __init__(self, step_dict: Dict[str, Step], is_ordered: bool = False):
        # TODO: What happens with anonymous steps in here?

        # This is to avoid an extra call for ordering the steps if constructing through from_params.
        # TODO: Perhaps there's a better way to do this? inspect to check if the caller is from_params?
        if not is_ordered:
            self.parsed_steps = {step.name: step for step in self.ordered_steps(step_dict)}
        else:
            self.parsed_steps = {step_name: step for step_name, step in step_dict.items()}

        # Sanity-check the graph
        self._sanity_check()

    @classmethod
    def _check_unsatisfiable_dependencies(cls, dependencies: Dict[str, Set[str]]) -> None:
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

    @classmethod
    def _get_ordered_steps(cls, dependencies: Dict[str, Set[str]]) -> List[str]:
        done: Set[str] = set()
        todo = list(dependencies.keys())
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
                    "You probably have a circular reference between the steps, "
                    "Or a missing dependency."
                )
            todo = new_todo
        del dependencies
        del done
        del todo
        return ordered_steps

    def _sanity_check(self) -> None:
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

    @classmethod
    def from_params(cls: Type["StepGraph"], params: Dict[str, Params]) -> "StepGraph":  # type: ignore[override]
        # Determine the order in which to create steps so that all dependent steps are available when we need them.
        # This algorithm for resolving step dependencies is O(n^2). Since we're
        # anticipating the number of steps in a single config to be in the dozens at most (#famouslastwords),
        # we choose simplicity over cleverness.
        dependencies = {
            step_name: cls._find_step_dependencies(step_params)
            for step_name, step_params in params.items()
        }
        cls._check_unsatisfiable_dependencies(dependencies)

        # We need ordered dependencies to construct the steps with refs.
        ordered_steps = cls._get_ordered_steps(dependencies)

        # Parse the steps
        step_dict: Dict[str, Step] = {}
        for step_name in ordered_steps:
            step_params = params.pop(step_name)
            if step_name in step_dict:
                raise ConfigurationError(f"Duplicate step name {step_name}")

            step_params = cls._replace_step_dependencies(step_params, step_dict)
            step_dict[step_name] = Step.from_params(step_params, step_name=step_name)

        return cls(step_dict, is_ordered=True)

    def get_sub_graph(self, step_name: str) -> "StepGraph":
        if step_name not in self.parsed_steps:
            raise KeyError(
                f"{step_name} is not a part of this StepGraph. "
                f"Available steps are: {list(self.parsed_steps.keys())}"
            )
        step_dict = {dep.name: dep for dep in self.parsed_steps[step_name].recursive_dependencies}
        step_dict[step_name] = self.parsed_steps[step_name]
        return StepGraph(step_dict)

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

    @classmethod
    def ordered_steps(cls, step_dict: Dict[str, Step]) -> List[Step]:
        """
        Returns the steps in this step graph in an order that can be executed one at a time.

        This does not take into account which steps may be cached. It simply returns an executable
        order of steps.
        """
        dependencies = {
            step_name: set([dep.name for dep in step.dependencies])
            for step_name, step in step_dict.items()
        }
        result: List[Step] = []
        for step_name in cls._get_ordered_steps(dependencies):
            result.append(step_dict[step_name])
        return result

    def find_uncacheable_leaf_steps(self) -> Set[Step]:
        interior_steps: Set[Step] = set()
        for _, step in self.parsed_steps.items():
            for dependency in step.dependencies:
                interior_steps.add(dependency)
        uncacheable_leaf_steps = {
            step for step in set(self.values()) - interior_steps if not step.cache_results
        }
        return uncacheable_leaf_steps

    @classmethod
    def from_file(cls, filename: PathOrStr) -> "StepGraph":
        params = Params.from_file(filename)
        for package_name in params.pop("include_package", []):
            import_extra_module(package_name)
        return StepGraph.from_params(params.pop("steps", keep_as_dict=True))

    @classmethod
    def _to_config(cls, o: Any):
        # TODO: get rid of repeated logic.
        if isinstance(o, (list, tuple, set)):
            return o.__class__(cls._to_config(i) for i in o)
        elif isinstance(o, dict):
            return {key: cls._to_config(value) for key, value in o.items()}
        elif isinstance(o, Step):
            return {"type": "ref", "ref": o.name}
        elif o is not None and not isinstance(o, (bool, str, int, float)):
            raise ValueError(o)
        return o

    def to_file(self, filename: PathOrStr) -> None:
        step_dict = {}
        for step_name, step in self.parsed_steps.items():
            if step.config is not None:
                step_dict[step_name] = {
                    key: self._to_config(value) for key, value in step.config.items()
                }
            else:
                # This will not contain "type". Currently, this blocks us from running this for graphs
                # not constructed using configs.
                # return step.to_params().as_dict()
                # TODO: be more informative with the error.
                raise RuntimeError(f"Could not construct the parameters for {step_name}.")

        params = Params({"steps": step_dict})
        params.to_file(filename)
