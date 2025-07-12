from aim import Repo

from track.proxy.auth_proxy import start_proxy


class ProxyRepoFactory:
    """Creates Repo objects that use an API key."""

    def __init__(self, repo_url: str, api_key: str):
        self.repo_url = repo_url
        self.api_key = api_key

    def __call__(self) -> Repo:
        # Aim itself doesn't support authentication (at least, not in the community edition)
        # Start a proxy server that adds an auth token, and point the Repo at that instead
        # The server must also be wrapped with middleware to validate the auth token (see auth_wrapper)
        proxy = start_proxy(self.repo_url, self.api_key, listen_port=0)
        repo = Repo(f'aim://{proxy.listen_host}:{proxy.listen_port}')
        return repo
