import json
from pathlib import Path

from tango.step_info import StepInfo
from test_fixtures.package.steps import FloatStep


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
