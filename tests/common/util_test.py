import os
import time
from pathlib import Path

import pytest
from flaky import flaky

from tango.common.testing import TangoTestCase
from tango.common.util import (
    could_be_class_name,
    find_integrations,
    find_submodules,
    resolve_module_name,
    threaded_generator,
)


class TestResolveModuleName(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        self._work_dir_restore = os.getcwd()
        os.chdir(self.TEST_DIR)

    def teardown_method(self):
        super().teardown_method()
        os.chdir(self._work_dir_restore)

    def test_with_package_init_file(self):
        path = Path("fake_package/fake_module/__init__.py")
        (self.TEST_DIR / path.parent).mkdir(parents=True)
        open(path, "w").close()
        open(path.parent.parent / "__init__.py", "w").close()
        assert resolve_module_name(str(path)) == ("fake_package.fake_module", Path("."))

    def test_with_submodule(self):
        path = Path("fake_package/fake_module")
        (self.TEST_DIR / path).mkdir(parents=True)
        open(path / "__init__.py", "w").close()
        open(path.parent / "__init__.py", "w").close()
        assert resolve_module_name(str(path)) == ("fake_package.fake_module", Path("."))

    def test_with_module_in_child_directory(self):
        path = Path("some_dir/fake_module.py")
        (self.TEST_DIR / path.parent).mkdir(parents=True)
        open(path, "w").close()
        assert resolve_module_name(str(path)) == ("fake_module", Path("./some_dir"))


def test_find_submodules():
    assert "tango.version" in set(find_submodules())
    assert "tango.common.registrable" in set(find_submodules())
    assert "tango.common" in set(find_submodules(recursive=False))
    assert "tango.common.registrable" not in set(find_submodules(recursive=False))
    assert "tango.integrations.torch" in set(find_submodules("integrations"))
    assert "tango.integrations.torch" not in set(find_submodules(exclude={"tango.integrations*"}))


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


@flaky(max_runs=3)
def test_threaded_generator():
    def generate_slowly():
        for i in range(10):
            yield i
            time.sleep(0.1)

    start = time.time()
    for i in threaded_generator(generate_slowly()):
        time.sleep(0.1)
    end = time.time()

    assert end - start < 11
