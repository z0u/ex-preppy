import logging

from aim.ext.transport.client import Client

log = logging.getLogger(__name__)

# Say we want to provide some headers for Aim to use when it makes its requests to the server.
#
# But Repo doesn't accept a Client, and Client doesn't accept headers. There are two problems:
#
# - Client initiates communication inside __init__, before we have a chance to set headers
# - Repo creates its own Client, so even if we could create an instance with headers, we couldn't use it.
#
# So we need to create our own version of Client.


class AuthorizedClient(Client):
    """A patched Aim client that adds an Authorization header."""

    BEARER_TOKEN: str | None = None

    @property
    def request_headers(self) -> dict:
        return self._request_headers

    @request_headers.setter
    def request_headers(self, headers: dict):
        token = type(self).BEARER_TOKEN
        if token and 'Authorization' not in headers:
            headers['Authorization'] = f'Bearer {token}'
        self._request_headers = headers


def patch_aim_client(bearer_token: str | None):
    """Monkey-patch the Aim client for authentication."""
    import aim.sdk.repo

    if bearer_token:
        if aim.sdk.repo.Client != AuthorizedClient:
            log.info('Patching Aim Client to use Basic auth')
            aim.sdk.repo.Client = AuthorizedClient
        if AuthorizedClient.BEARER_TOKEN != bearer_token:
            log.info('Setting bearer token')
            AuthorizedClient.BEARER_TOKEN = bearer_token
    else:
        if aim.sdk.repo.Client != Client:
            log.info('Restoring original Aim Client')
            AuthorizedClient.BEARER_TOKEN = None
            aim.sdk.repo.Client = Client
