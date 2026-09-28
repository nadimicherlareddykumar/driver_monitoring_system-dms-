import pytest

from dms_utils.actions import (
    ACTIONS,
    ACTIONS_EXTENDED,
    get_num_classes,
)


class TestActions:
    def test_default_actions(self):
        assert ACTIONS["phonecall"] == 0
        assert ACTIONS["texting"] == 1
        assert len(ACTIONS) == 2

    def test_extended_actions(self):
        assert ACTIONS_EXTENDED["phonecall"] == 0
        assert ACTIONS_EXTENDED["texting"] == 1
        assert ACTIONS_EXTENDED["smoking"] == 2
        assert ACTIONS_EXTENDED["drinking"] == 3
        assert ACTIONS_EXTENDED["eating"] == 4
        assert len(ACTIONS_EXTENDED) == 8


class TestGetNumClasses:
    def test_default_num_classes(self):
        assert get_num_classes(extended=False) == 2

    def test_extended_num_classes(self):
        assert get_num_classes(extended=True) == 8
