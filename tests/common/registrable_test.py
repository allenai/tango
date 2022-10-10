import pytest

from tango.common.exceptions import ConfigurationError
from tango.common.registrable import Registrable
from tango.common.testing import TangoTestCase
from tango.step import Step


class TestRegistrable(TangoTestCase):
    def test_basic_functionality(self):
        class MockBaseClass(Registrable):
            pass

        assert "mock-1" not in MockBaseClass.list_available()

        @MockBaseClass.register("mock-1")
        class MockSubclass1(MockBaseClass):
            pass

        assert MockBaseClass in Registrable._registry
        assert MockBaseClass.by_name("mock-1") == MockSubclass1

        # Verify that registering under a name that already exists
        # causes a ConfigurationError.
        with pytest.raises(ConfigurationError):

            @MockBaseClass.register("mock-1")
            class MockAlternate(MockBaseClass):
                pass

        # Registering under a name that already exists should overwrite
        # if exist_ok=True.
        @MockBaseClass.register("mock-1", exist_ok=True)
        class MockAlternate2(MockBaseClass):
            pass

        assert MockBaseClass.by_name("mock-1") == MockAlternate2

        # Test that we get a suggestion when the name is close.
        with pytest.raises(ConfigurationError) as exc:
            MockBaseClass.by_name("mock_1")
            assert "did you mean 'mock-1'?" in str(exc.value)

    def test_registering_step_by_reserved_name(self):
        with pytest.raises(ConfigurationError, match="cannot use the name 'ref'"):

            @Step.register("ref")
            class BadStep(Step):
                pass

    def test_search_modules(self):
        Step.search_modules("foo-bar-baz-non-existent")
