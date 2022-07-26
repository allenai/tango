import json
from pathlib import Path
from typing import Any

from tango.common.testing.steps import FloatStep
from tango.step import Step
from tango.step_graph import StepGraph
from tango.step_info import StepInfo


def test_step_info():
    step = FloatStep(step_name="float", result=1.0)
    step_info = StepInfo.new_from_step(step)

    # Check Git metadata.
    if (Path.cwd() / ".git").exists():
        assert step_info.environment.git is not None
        assert step_info.environment.git.commit is not None
        assert step_info.environment.git.remote is not None
        assert "allenai/tango" in step_info.environment.git.remote

    # Check pip requirements.
    assert step_info.environment.packages is not None

    # Test serialization / deserialization.
    serialized = json.dumps(step_info.to_json_dict())
    deserialized = StepInfo.from_json_dict(json.loads(serialized))
    assert deserialized == step_info


def test_step_info_with_step_dependency():
    """Checks that the StepInfo config is not parsed to a Step if it has dependencies on upstream steps"""

    @Step.register("foo")
    class FooStep(Step):
        def run(self, bar: Any) -> str:
            return "foo" + bar

    @Step.register("bar")
    class BarStep(Step):
        def run(self) -> str:  # type: ignore
            return "Hey!"

    graph = StepGraph.from_params(
        {
            "foo": {
                "type": "foo",
                "bar": {"type": "ref", "ref": "bar"},
            },
            "bar": {
                "type": "bar",
            },
        }
    )
    step = graph["foo"]
    step_info = StepInfo.new_from_step(step)

    step_info_json = json.dumps(step_info.to_json_dict())
    step_info = StepInfo.from_json_dict(json.loads(step_info_json))
    isinstance(step_info.config, dict)
