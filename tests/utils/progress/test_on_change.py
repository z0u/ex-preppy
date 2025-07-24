"""Test for OnChangeDict and OnChangeList."""

from unittest.mock import Mock

import pytest
from src.utils.progress.on_change import OnChangeDict, OnChangeList


class TestOnChangeDict:
    """Test the OnChangeDict."""

    @pytest.fixture
    def on_change_mock(self):
        """Return a mock for the on_change callback."""
        return Mock()

    @pytest.fixture
    def observable_dict(self, on_change_mock: Mock) -> OnChangeDict[str, int]:
        """Return an OnChangeDict with a mock callback."""
        obs_dict = OnChangeDict[str, int]({'a': 1, 'b': 2})
        obs_dict._on_change = on_change_mock
        return obs_dict

    def test_setitem_new_key(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that __setitem__ with a new key calls on_change once."""
        observable_dict['c'] = 3
        on_change_mock.assert_called_once()

    def test_setitem_existing_key(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that __setitem__ with an existing key calls on_change once."""
        observable_dict['a'] = 5
        on_change_mock.assert_called_once()

    def test_delitem(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that __delitem__ calls on_change once."""
        del observable_dict['a']
        on_change_mock.assert_called_once()

    def test_clear(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that clear calls on_change once."""
        observable_dict.clear()
        on_change_mock.assert_called_once()

    def test_pop(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that pop calls on_change once."""
        observable_dict.pop('a')
        on_change_mock.assert_called_once()

    def test_union(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that the union operator calls on_change once."""
        observable_dict |= {'c': 3}
        on_change_mock.assert_called_once()

    def test_popitem(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that popitem calls on_change once."""
        observable_dict.popitem()
        on_change_mock.assert_called_once()

    def test_setdefault_new_key(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that setdefault with a new key calls on_change once."""
        observable_dict.setdefault('c', 3)
        on_change_mock.assert_called_once()

    def test_setdefault_existing_key(self, observable_dict: OnChangeDict[str, int], on_change_mock: Mock):
        """Test that setdefault with an existing key does not call on_change."""
        observable_dict.setdefault('a', 5)
        on_change_mock.assert_not_called()


class TestOnChangeList:
    """Test the OnChangeList."""

    @pytest.fixture
    def on_change_mock(self):
        """Return a mock for the on_change callback."""
        return Mock()

    @pytest.fixture
    def observable_list(self, on_change_mock: Mock) -> OnChangeList[int]:
        """Return an OnChangeList with a mock callback."""
        obs_list = OnChangeList[int]([1, 2, 3])
        obs_list._on_change = on_change_mock
        return obs_list

    def test_setitem(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that __setitem__ calls on_change once."""
        observable_list[0] = 100
        on_change_mock.assert_called_once()

    def test_delitem(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that __delitem__ calls on_change once."""
        del observable_list[0]
        on_change_mock.assert_called_once()

    def test_iadd(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that __iadd__ calls on_change once (via extend)."""
        observable_list += [4, 5]
        on_change_mock.assert_called_once()

    def test_append(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that append calls on_change once."""
        observable_list.append(4)
        on_change_mock.assert_called_once()

    def test_extend(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that extend calls on_change once."""
        observable_list.extend([4, 5])
        on_change_mock.assert_called_once()

    def test_insert(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that insert calls on_change once."""
        observable_list.insert(1, 100)
        on_change_mock.assert_called_once()

    def test_pop(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that pop calls on_change once."""
        observable_list.pop()
        on_change_mock.assert_called_once()

    def test_remove(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that remove calls on_change once."""
        observable_list.remove(2)
        on_change_mock.assert_called_once()

    def test_clear(self, observable_list: OnChangeList[int], on_change_mock: Mock):
        """Test that clear calls on_change once."""
        observable_list.clear()
        on_change_mock.assert_called_once()
