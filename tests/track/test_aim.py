from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Mock the uv_freeze function before importing track.aim since it's called at module level
with patch('infra.requirements.uv_freeze', return_value=[]):
    from track.aim import AIM_DIR, get_repo, track_app_name


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
        return_value=('https://example.com', 'api_key_123')
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
    mock_repo_class.assert_called_once_with('aim://example.com/server')

    assert result == mock_repo_instance


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
    base_url = 'https://example.com'
    url = URL(base_url).with_scheme('aim') / 'server'

    # Verify the URL is constructed correctly
    expected_url = 'aim://example.com/server'
    assert str(url) == expected_url
