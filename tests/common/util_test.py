import pytest

from tango.common.util import could_be_class_name, find_integrations, find_submodules


def test_find_submodules():
    assert "tango.version" in set(find_submodules())
    assert "tango.common.registrable" in set(find_submodules())
    assert "tango.common" in set(find_submodules(recursive=False))
    assert "tango.common.registrable" not in set(find_submodules(recursive=False))
    assert "tango.integrations.torch" in set(find_submodules("integrations"))
    assert "tango.integrations.torch" not in set(find_submodules(exclude="tango.integrations*"))


def test_find_integrations():
    integrations = set(find_integrations())
    assert "tango.integrations.torch" in integrations
    assert "tango.integrations.torch.format" not in integrations


@pytest.mark.parametrize(
    "name, result",
    [
        ("", False),
        ("foo.Bar", True),
        ("foo.Bar.", False),
        ("1foo.Bar", False),
    ],
)
def test_could_be_class_name(name: str, result: bool):
    assert could_be_class_name(name) is result
