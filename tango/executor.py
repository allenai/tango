from abc import abstractmethod
from typing import List, Tuple, Iterable, Any, Dict, Optional

import click

from tango.common.params import Params
from tango.common.registrable import Registrable
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_graph import StepGraph, StepStub


class Executor(Registrable):
    """
    An ``Executor`` is :class:`~tango.common.registrable.Registrable` class that is
    responsible for running steps and returning them with their result.

    Subclasses should implement :meth:`execute_step_group()`.
    """

    default_implementation = "simple"

    def __init__(self, include_package: Optional[List[str]] = None) -> None:
        self.include_package = include_package

    def execute_step_graph(self, step_graph: StepGraph, step_cache: StepCache) -> List[Step]:
        """
        Execute an entire :class:`tango.step_graph.StepGraph`.
        """
        # Keeps track of all steps that we've ran so far by name, and stores the results
        # for ones that can't be cached.
        executed: Dict[str, Tuple[Step, Any]] = {}
        remaining: List[StepStub] = list(step_graph)
        group: List[Step] = []
        out: List[Step] = []

        def run_group():
            # Gather all steps that need to be ran (no cache hit).
            needed: List[Step] = []
            for step in group:
                if step not in step_cache:
                    needed.append(step)
                else:
                    executed[step.name] = (step, None)
                    click.echo(
                        click.style("✓ Found output for ", fg="green")
                        + click.style(f'"{step.name}"', bold=True, fg="green")
                        + click.style(" in cache", fg="green")
                    )
            if needed:
                # Run the needed ones.
                for step, result in self.execute_step_group(needed, step_cache):
                    # Add to cache or keep result in `executed`.
                    if step.cache_results and step not in step_cache:
                        step_cache[step] = result
                        executed[step.name] = (step, None)
                    else:
                        executed[step.name] = (step, result)

        while remaining:
            next_step_ready = True
            for ref in remaining[0].dependencies:
                if ref not in executed:
                    # Still has dependencies that need to be ran first.
                    next_step_ready = False
                    break

            if not next_step_ready:
                # Run the current group.
                assert group
                run_group()
                out.extend(group)
                group = []

            # Materialize the next step.
            next_up = remaining.pop(0)
            config = self._replace_refs_with_results(next_up.config, executed, step_cache)
            step = Step.from_params(Params(config), step_name=next_up.name)
            step._executor = self
            group.append(step)

        # Finish up last group.
        if group:
            run_group()
            out.extend(group)
        return out

    @abstractmethod
    def execute_step_group(
        self, step_group: List[Step], step_cache: StepCache
    ) -> Iterable[Tuple[Step, Any]]:
        """
        Execute all steps in the group, returning them with their results.

        The executor can assume that all steps in the group are independent (none of them
        depend on the result of any other step in the group), so they can be ran
        in any order or even in parallel.
        """
        raise NotImplementedError

    @staticmethod
    def _replace_refs_with_results(
        o: Any, executed: Dict[str, Tuple[Step, Any]], step_cache: StepCache
    ) -> Any:
        if isinstance(o, list):
            return [Executor._replace_refs_with_results(x, executed, step_cache) for x in o]
        elif isinstance(o, tuple):
            return tuple(Executor._replace_refs_with_results(list(o), executed, step_cache))
        elif isinstance(o, set):
            return set(Executor._replace_refs_with_results(list(o), executed, step_cache))
        elif isinstance(o, dict):
            if set(o.keys()) == {"type", "ref"}:
                if o["ref"] in executed:
                    step, result = executed[o["ref"]]
                    if result is not None:
                        return result
                    else:
                        return step_cache[step]
                else:
                    raise ValueError(f"result for step '{o['ref']}' could not be found!")
            else:
                return {
                    k: Executor._replace_refs_with_results(v, executed, step_cache)
                    for k, v in o.items()
                }
        elif isinstance(o, (str, bool, int, float)):
            return o
        else:
            raise ValueError(o)


@Executor.register("simple")
class SimpleExecutor(Executor):
    """
    A simple :class:`Executor` that just runs all steps locally one at a time.
    """

    def execute_step_group(
        self, step_group: List[Step], step_cache: StepCache
    ) -> Iterable[Tuple[Step, Any]]:
        for step in step_group:
            click.echo(
                click.style("● Starting run for ", fg="blue")
                + click.style(f'"{step.name}"', bold=True, fg="blue")
            )
            result = step.result(step_cache)
            click.echo(
                click.style("✓ Finished run for ", fg="green")
                + click.style(f'"{step.name}"', bold=True, fg="green")
            )
            yield step, result
