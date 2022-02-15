from tango.common.testing import TangoTestCase
from tango.step import Step
from tango.step_graph import StepGraph


@Step.register("add_numbers")
class AddNumbers(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a_number: int, b_number: int) -> int:  # type: ignore
        return a_number + b_number


class TestStepGraph(TangoTestCase):
    def test_ordered_steps(self):
        step_graph = StepGraph(
            {
                "step1": {
                    "type": "add_numbers",
                    "a_number": 2,
                    "b_number": 3,
                },
                "step2": {
                    "type": "add_numbers",
                    "a_number": {"type": "ref", "ref": "step1"},
                    "b_number": 5,
                },
                "step3": {
                    "a_number": 3,
                    "b_number": 1,
                },
            }
        )

        result = step_graph.ordered_steps()
        assert [res.name for res in result] == ["step1", "step2", "step3"]

    def test_from_file(self):
        step_graph = StepGraph.from_file(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet")
        assert "hello" in step_graph
        assert "hello_world" in step_graph
