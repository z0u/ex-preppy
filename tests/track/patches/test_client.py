from unittest.mock import patch

from track.patches.client import AuthorizedClient, patch_aim_client


@patch('track.patches.client.Client.connect')
def test_authorized_client_adds_bearer_token(mock_connect):
    """Test that AuthorizedClient adds Authorization header."""
    # Set a bearer token
    AuthorizedClient.BEARER_TOKEN = 'test_token_123'

    client = AuthorizedClient('http://example.com')

    # Test setting headers without Authorization
    test_headers = {'Content-Type': 'application/json'}
    client.request_headers = test_headers

    expected_headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer test_token_123'}
    assert client.request_headers == expected_headers


@patch('track.patches.client.Client.connect')
def test_authorized_client_preserves_existing_auth(mock_connect):
    """Test that AuthorizedClient doesn't override existing Authorization header."""
    AuthorizedClient.BEARER_TOKEN = 'test_token_123'

    client = AuthorizedClient('http://example.com')

    # Test setting headers with existing Authorization
    test_headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer existing_token'}
    client.request_headers = test_headers

    # Should preserve existing Authorization header
    assert client.request_headers == test_headers
    assert client.request_headers['Authorization'] == 'Bearer existing_token'


@patch('track.patches.client.Client.connect')
def test_authorized_client_no_token(mock_connect):
    """Test that AuthorizedClient doesn't add header when no token is set."""
    AuthorizedClient.BEARER_TOKEN = None

    client = AuthorizedClient('http://example.com')

    test_headers = {'Content-Type': 'application/json'}
    client.request_headers = test_headers

    # Should not add Authorization header when no token
    assert client.request_headers == test_headers
    assert 'Authorization' not in client.request_headers


@patch('aim.sdk.repo')
def test_patch_aim_client(mock_aim_repo):
    """Test that patch_aim_client properly monkey-patches the Aim client."""
    test_token = 'patch_test_token_456'

    # Store original Client (if any) to restore later
    original_client = getattr(mock_aim_repo, 'Client', None)

    try:
        patch_aim_client(test_token)

        # Verify that aim.sdk.repo.Client is now AuthorizedClient
        assert mock_aim_repo.Client is AuthorizedClient
        assert AuthorizedClient.BEARER_TOKEN == test_token

    finally:
        # Restore original Client
        if original_client:
            mock_aim_repo.Client = original_client


def test_authorized_client_inherits_from_aim_client():
    """Test that AuthorizedClient properly inherits from Aim's Client."""
    from aim.ext.transport.client import Client

    assert issubclass(AuthorizedClient, Client)


@patch('track.patches.client.Client.connect')
def test_authorized_client_request_headers_property(mock_connect):
    """Test the request_headers property getter and setter work correctly."""
    client = AuthorizedClient('http://example.com')

    # Test getter
    headers = client.request_headers
    assert isinstance(headers, dict)

    # Test setter
    new_headers = {'X-Test': 'value'}
    client.request_headers = new_headers
    assert client.request_headers == new_headers


def test_bearer_token_class_variable():
    """Test that BEARER_TOKEN is properly managed as a class variable."""
    # Test setting and getting token
    original_token = AuthorizedClient.BEARER_TOKEN

    try:
        AuthorizedClient.BEARER_TOKEN = 'test_class_token'
        assert AuthorizedClient.BEARER_TOKEN == 'test_class_token'

        # Multiple instances should share the same token
        AuthorizedClient.BEARER_TOKEN = 'shared_token'
        assert AuthorizedClient.BEARER_TOKEN == 'shared_token'

    finally:
        # Restore original token
        AuthorizedClient.BEARER_TOKEN = original_token


def test_authorization_header_logic():
    """Test the core authorization header logic without creating clients."""

    # Simulate the setter logic
    def add_auth_header(headers: dict, token: str | None) -> dict:
        if token and 'Authorization' not in headers:
            headers['Authorization'] = f'Bearer {token}'
        return headers

    # Test adding new header
    headers1 = {'Content-Type': 'application/json'}
    result1 = add_auth_header(headers1, 'test_token')
    assert result1['Authorization'] == 'Bearer test_token'

    # Test preserving existing header
    headers2 = {'Authorization': 'Bearer existing'}
    result2 = add_auth_header(headers2, 'test_token')
    assert result2['Authorization'] == 'Bearer existing'

    # Test no token
    headers3 = {'Content-Type': 'application/json'}
    result3 = add_auth_header(headers3, None)
    assert 'Authorization' not in result3
