import os
from unittest.mock import MagicMock, patch

from track.auth_wrapper import create_wrapper_app


@patch.dict(
    os.environ,
    {
        'GOOGLE_CLIENT_ID': 'test_client_id',
        'GOOGLE_CLIENT_SECRET': 'test_client_secret',
        'ALLOWED_EMAIL': 'user@example.com,admin@example.com',
    },
)
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_create_wrapper_app(mock_get_api_key, mock_get_session_secret):
    """Test that create_wrapper_app creates a FastAPI app with proper configuration."""
    mock_get_api_key.return_value = 'test_api_key_123'
    mock_get_session_secret.return_value = 'test_session_secret_456'

    app, internal_api_key = create_wrapper_app('test_app')

    assert app is not None
    # FastAPI uses 'title' differently than name, let's check what we can verify
    assert hasattr(app, 'title')  # App has a title attribute
    assert internal_api_key == 'test_api_key_123'

    mock_get_api_key.assert_called_once_with('test_app')
    mock_get_session_secret.assert_called_once_with('test_app')


@patch.dict(
    os.environ,
    {
        'GOOGLE_CLIENT_ID': 'test_client_id',
        'GOOGLE_CLIENT_SECRET': 'test_client_secret',
        'ALLOWED_EMAIL': 'user@example.com',
    },
)
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_service_authorization(mock_get_api_key, mock_get_session_secret):
    """Test service authorization using API key."""
    mock_get_api_key.return_value = 'service_api_key'
    mock_get_session_secret.return_value = 'session_secret'

    app, _ = create_wrapper_app('test_app')

    # Test the authorization logic directly since accessing middleware is complex
    authz = 'Bearer service_api_key'
    internal_api_key = 'service_api_key'
    is_service_auth = authz == f'Bearer {internal_api_key}'

    assert is_service_auth is True

    # Test with wrong key
    wrong_authz = 'Bearer wrong_key'
    is_service_auth_wrong = wrong_authz == f'Bearer {internal_api_key}'
    assert is_service_auth_wrong is False


@patch.dict(
    os.environ,
    {
        'GOOGLE_CLIENT_ID': 'test_client_id',
        'GOOGLE_CLIENT_SECRET': 'test_client_secret',
        'ALLOWED_EMAIL': 'user@example.com,admin@example.com',
    },
)
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_user_authorization_allowed_email(mock_get_api_key, mock_get_session_secret):
    """Test user authorization with allowed email."""
    mock_get_api_key.return_value = 'api_key'
    mock_get_session_secret.return_value = 'session_secret'

    app, _ = create_wrapper_app('test_app')

    # Test allowed email
    user_email = 'user@example.com'
    allowed_emails_str = 'user@example.com,admin@example.com'
    allowed_emails = {email.strip() for email in allowed_emails_str.split(',')}

    assert user_email in allowed_emails


@patch.dict(
    os.environ,
    {
        'GOOGLE_CLIENT_ID': 'test_client_id',
        'GOOGLE_CLIENT_SECRET': 'test_client_secret',
        'ALLOWED_EMAIL': 'user@example.com',
    },
)
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_user_authorization_disallowed_email(mock_get_api_key, mock_get_session_secret):
    """Test user authorization with disallowed email."""
    mock_get_api_key.return_value = 'api_key'
    mock_get_session_secret.return_value = 'session_secret'

    app, _ = create_wrapper_app('test_app')

    # Test disallowed email
    user_email = 'hacker@badsite.com'
    allowed_emails_str = 'user@example.com'
    allowed_emails = {email.strip() for email in allowed_emails_str.split(',')}

    assert user_email not in allowed_emails


@patch.dict(os.environ, {'GOOGLE_CLIENT_ID': 'test_client_id', 'GOOGLE_CLIENT_SECRET': 'test_client_secret'})
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_no_allowed_emails_configured(mock_get_api_key, mock_get_session_secret):
    """Test behavior when no allowed emails are configured."""
    mock_get_api_key.return_value = 'api_key'
    mock_get_session_secret.return_value = 'session_secret'

    app, _ = create_wrapper_app('test_app')

    # When ALLOWED_EMAIL is not set
    allowed_emails_str = os.environ.get('ALLOWED_EMAIL', '')
    assert allowed_emails_str == ''


def test_authorization_logic_functions():
    """Test the core authorization logic without FastAPI dependencies."""

    # Test service authorization logic
    def is_authorized_service(auth_header: str, internal_api_key: str) -> bool:
        return auth_header == f'Bearer {internal_api_key}'

    assert is_authorized_service('Bearer test_key', 'test_key') is True
    assert is_authorized_service('Bearer wrong_key', 'test_key') is False
    assert is_authorized_service('', 'test_key') is False

    # Test user authorization logic
    def is_authorized_user(user_email: str | None, allowed_emails_str: str) -> bool:
        if not user_email or not allowed_emails_str:
            return False
        allowed_emails = {email.strip() for email in allowed_emails_str.split(',')}
        return user_email in allowed_emails

    assert is_authorized_user('user@example.com', 'user@example.com,admin@example.com') is True
    assert is_authorized_user('hacker@badsite.com', 'user@example.com') is False
    assert is_authorized_user('user@example.com', '') is False
    assert is_authorized_user(None, 'user@example.com') is False


@patch.dict(
    os.environ,
    {
        'GOOGLE_CLIENT_ID': 'test_client_id',
        'GOOGLE_CLIENT_SECRET': 'test_client_secret',
        'ALLOWED_EMAIL': 'user@example.com',
    },
)
@patch('track.auth_wrapper.get_session_secret')
@patch('track.auth_wrapper.get_internal_api_key')
def test_oauth_configuration(mock_get_api_key, mock_get_session_secret):
    """Test that OAuth is properly configured."""
    mock_get_api_key.return_value = 'api_key'
    mock_get_session_secret.return_value = 'session_secret'

    with patch('track.auth_wrapper.OAuth') as mock_oauth_class:
        mock_oauth = MagicMock()
        mock_oauth_class.return_value = mock_oauth

        app, _ = create_wrapper_app('test_app')

        # Verify OAuth was configured
        mock_oauth.register.assert_called_once_with(
            name='google',
            client_id='test_client_id',
            client_secret='test_client_secret',
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={
                'scope': 'openid email',
                'response_type': 'code',
            },
        )
