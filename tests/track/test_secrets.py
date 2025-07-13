from unittest.mock import MagicMock, patch

from track.secrets import get_session_secret, get_internal_api_key, _strong_secret_factory


def test_strong_secret_factory():
    """Test that secret factory generates strong secrets."""
    secret = _strong_secret_factory()

    assert isinstance(secret, str)
    assert len(secret) > 30  # URL-safe base64 encoding of 256 bits should be ~43 chars
    assert secret != _strong_secret_factory()  # Should be different each time


@patch('track.secrets._get_store')
def test_get_session_secret(mock_get_store):
    """Test session secret retrieval with mocked Modal Dict."""
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store

    # Test when key doesn't exist - should create new secret
    mock_store.__contains__.return_value = False
    mock_store.__getitem__.return_value = 'test_secret_123'

    result = get_session_secret('test_app')

    mock_get_store.assert_called_once_with('test_app')
    mock_store.put.assert_called_once()
    assert result == 'test_secret_123'


@patch('track.secrets._get_store')
def test_get_internal_api_key(mock_get_store):
    """Test internal API key retrieval with mocked Modal Dict."""
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store

    # Test when key exists - should return existing value
    mock_store.__contains__.return_value = True
    mock_store.__getitem__.return_value = 'existing_api_key_456'

    result = get_internal_api_key('test_app')

    mock_get_store.assert_called_once_with('test_app')
    mock_store.put.assert_not_called()  # Should not create new value
    assert result == 'existing_api_key_456'


def test_get_or_create_skip_if_exists():
    """Test that _get_or_create properly uses skip_if_exists."""
    from track.secrets import _get_or_create

    mock_store = MagicMock()

    # Test when key doesn't exist
    mock_store.__contains__.return_value = False
    mock_store.__getitem__.return_value = 'new_value'

    def test_factory():
        return 'factory_result'

    result = _get_or_create(mock_store, 'test_key', test_factory)

    mock_store.put.assert_called_once_with('test_key', 'factory_result', skip_if_exists=True)
    assert result == 'new_value'


@patch('track.secrets.modal.Dict')
def test_get_store_caching_real(mock_dict_class):
    """Test that _get_store properly caches store instances."""
    from track.secrets import _get_store

    # Clear the cache first to ensure clean test
    _get_store.cache_clear()

    mock_store1 = MagicMock()
    mock_store2 = MagicMock()
    mock_dict_class.from_name.side_effect = [mock_store1, mock_store2]

    # Call twice with same app name - should use cache
    store1 = _get_store('same_app')
    store2 = _get_store('same_app')

    assert store1 is store2  # Should be the same object due to caching
    assert mock_dict_class.from_name.call_count == 1  # Only called once

    # Call with different app name - should create new store
    store3 = _get_store('different_app')
    assert store3 is not store1
    assert mock_dict_class.from_name.call_count == 2  # Called twice now
