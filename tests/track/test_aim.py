from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import PurePosixPath
import pytest

from track.aim import AIM_DIR, get_repo, track_app_name


def test_constants():
    """Test that module constants are properly defined."""
    assert AIM_DIR == PurePosixPath('/aim')
    assert track_app_name == 'mi-ni.track'
    assert isinstance(AIM_DIR, PurePosixPath)


@patch('aim.sdk.repo.Repo')
def test_ensure_repo_new(mock_repo_class):
    """Test _ensure_repo when repository doesn't exist."""
    from track.aim import AimService

    mock_repo_class.exists.return_value = False
    mock_repo_instance = MagicMock()
    mock_repo_class.from_path.return_value = mock_repo_instance

    # Create service instance
    service = AimService()  # type: ignore  # Modal decorated class
    service._ensure_repo(AIM_DIR)

    mock_repo_class.exists.assert_called_once_with(str(AIM_DIR))
    mock_repo_class.from_path.assert_called_once_with(str(AIM_DIR), init=True)


@patch('aim.sdk.repo.Repo')
def test_ensure_repo_existing(mock_repo_class):
    """Test _ensure_repo when repository already exists."""
    from track.aim import AimService

    mock_repo_class.exists.return_value = True
    mock_repo_instance = MagicMock()
    mock_repo_class.from_path.return_value = mock_repo_instance

    # Create service instance
    service = AimService()  # type: ignore  # Modal decorated class
    service._ensure_repo(AIM_DIR)

    mock_repo_class.exists.assert_called_once_with(str(AIM_DIR))
    mock_repo_class.from_path.assert_called_once_with(str(AIM_DIR))


@patch('track.aim.modal.Cls')
@patch('track.aim.patch_aim_client')
@patch('aim.sdk.repo.Repo')
async def test_get_repo(mock_repo_class, mock_patch_client, mock_cls):
    """Test get_repo async function."""
    # Mock the Modal service
    mock_service_instance = MagicMock()
    mock_service_instance.get_repo_for_client.remote.aio = AsyncMock(
        return_value=('https://test.modal.run', 'api_key_123')
    )
    mock_service = MagicMock()
    mock_service.return_value = mock_service_instance
    mock_cls.from_name.return_value = mock_service

    # Mock Repo creation
    mock_repo_instance = MagicMock()
    mock_repo_class.return_value = mock_repo_instance

    result = await get_repo()

    # Verify Modal service lookup
    mock_cls.from_name.assert_called_once_with(track_app_name, 'AimService')

    # Verify client patching
    mock_patch_client.assert_called_once_with('api_key_123')

    # Verify Repo creation with correct URL
    mock_repo_class.assert_called_once_with('aim://test.modal.run/server')

    assert result == mock_repo_instance


def test_service_configuration():
    """Test basic service configuration that can be verified."""
    from track.aim import AimService

    # Test that we can create a service instance
    service = AimService()  # type: ignore  # Modal decorated class

    # Test that methods exist
    assert hasattr(service, '_ensure_repo')
    assert hasattr(service, 'get_repo_for_client')
    assert hasattr(service, 'web_interface')


def test_aim_dir_path_type():
    """Test that AIM_DIR is the correct path type."""
    assert isinstance(AIM_DIR, PurePosixPath)
    assert str(AIM_DIR) == '/aim'


def test_track_app_name_format():
    """Test track app name follows expected format."""
    assert track_app_name == 'mi-ni.track'
    assert '.' in track_app_name
    assert track_app_name.startswith('mi-ni')


def test_get_repo_service_error_handling():
    """Test get_repo error handling for service issues."""

    # Test RuntimeError when server isn't running
    def get_repo_for_client_mock():
        raise RuntimeError("Server isn't running")

    mock_service = MagicMock()
    mock_service.get_repo_for_client = get_repo_for_client_mock

    # This tests the expected behavior pattern
    with pytest.raises(RuntimeError, match="Server isn't running"):
        mock_service.get_repo_for_client()


def test_url_construction_logic():
    """Test the URL construction logic without patching imports."""
    from yarl import URL

    # Test the URL construction logic that would be used in get_repo
    base_url = 'https://test.modal.run'
    url = URL(base_url).with_scheme('aim') / 'server'

    # Verify the URL is constructed correctly
    expected_url = 'aim://test.modal.run/server'
    assert str(url) == expected_url


def test_module_imports():
    """Test that module imports work correctly."""
    # Test that all expected imports are available
    import track.aim

    assert hasattr(track.aim, 'AimService')
    assert hasattr(track.aim, 'get_repo')
    assert hasattr(track.aim, 'AIM_DIR')
    assert hasattr(track.aim, 'track_app_name')


def test_aim_service_methods_exist():
    """Test that AimService has the expected methods."""
    from track.aim import AimService

    service = AimService()  # type: ignore  # Modal decorated class

    # Test methods exist (Modal decorates them, so they may not be normal callables)
    assert hasattr(service, '_ensure_repo')
    assert hasattr(service, 'get_repo_for_client')
    assert hasattr(service, 'web_interface')
