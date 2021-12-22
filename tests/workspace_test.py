from tango.workspace import StepInfo


class CustomError(Exception):
    def __init__(self, msg, foo, bar):
        super().__init__(msg)
        self.foo = foo
        self.bar = bar


def test_step_info_serde_with_custom_exception():
    """
    Tests that we can serialize and deserialize a ``StepInfo`` instance with
    a custom exception that has positional arguments.
    """
    step_info = StepInfo(
        unique_id="id-123",
        step_name=None,
        step_class_name=None,
        version=None,
        dependencies=set(),
        error=CustomError("Oh no!", "foo", "bar"),
    )
    dump = step_info.serialize()
    # Since 'CustomError' can't be pickled, we'll just have the string representation
    # of it.
    result = StepInfo.deserialize(dump)
    assert result.error == "CustomError('Oh no!')"
