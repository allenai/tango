from typing import Optional, Sequence

from beaker import Beaker

from tango.executor import Executor, ExecutorOutput
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import Workspace


@Executor.register("beaker")
class BeakerExecutor(Executor):
    """
    This is a :class:`~tango.executor.Executor` that runs steps on `Beaker`_.

    :param workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.

    .. tip::
        Registered as :class:`~tango.executor.Executor` under the name "beaker".
    """

    def __init__(
        self,
        workspace: Workspace,
        beaker_workspace: Optional[str] = None,
        include_package: Optional[Sequence[str]] = None,
        **kwargs
    ):
        super().__init__(workspace, include_package=include_package)
        if beaker_workspace is not None:
            kwargs["default_workspace"] = beaker_workspace
        self.beaker = Beaker.from_env(**kwargs)

    def execute_step(self, step: "Step") -> None:
        pass

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        pass
