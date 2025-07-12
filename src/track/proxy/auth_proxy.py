import atexit
import inspect
import logging

from proxy import Proxy

from track.proxy.plugins.auth import UpstreamAuthPlugin
from track.proxy.plugins.reverse import ReverseProxyPlugin

log = logging.getLogger(__name__)


class AuthProxy:
    """
    HTTP proxy that adds an Authorization header to requests.

    You can use it as a context manager, in which case it stops when exiting the
    context. Or you can call start and stop manually.
    """

    def __init__(
        self,
        target_addr: str,
        auth_token: str,
        listen_host: str = '127.0.0.1',  # loopback
        listen_port: int = 0,  # zero means "random"; check listen_port property after startup
    ):
        self.target_addr = target_addr
        self.listen_host = listen_host
        self._listen_port = listen_port
        self.auth_token = auth_token
        self.proxy = None

    @property
    def listen_port(self):
        return self.proxy.flags.port if self.proxy else self._listen_port

    @property
    def listen_addr(self):
        return f'http://{self.listen_host}:{self.listen_port}'

    def start(self):
        proxy = Proxy(
            # Plugins must be specified like this, or their custom flags (arguments) won't be known to proxy.py
            [
                '--plugins',
                get_identifier(ReverseProxyPlugin),
                get_identifier(UpstreamAuthPlugin),
                f'--hostname={self.listen_host}',
                f'--port={self.listen_port}',
                '--disable-http-proxy',  # Only reverse proxy
                '--enable-reverse-proxy',
                '--rewrite-host-header',
                f'--target-addr={self.target_addr}',
                f'--upstream-token={self.auth_token}',
            ],
        )
        self.proxy = proxy.__enter__()

        log.info(f'Auth proxy {self.listen_host}:{self.listen_port} -> {self.target_addr}')

    def stop(self, *args):
        if self.proxy:
            log.info('Stopping auth proxy')
            self.proxy.__exit__(*args)
            self.proxy = None


def get_identifier(ob) -> str:
    return inspect.getmodule(ob).__name__ + '.' + ob.__name__  # type: ignore


_auth_proxy_singleton: AuthProxy | None = None


def start_proxy(
    target_addr: str,
    auth_token: str,
    listen_host: str = '127.0.0.1',
    listen_port: int = 8080,
):
    """
    Start the auth proxy, replacing any existing instance.

    Since it replaces the existing instance, it can be re-run in a notebook cell.
    """
    global _auth_proxy_singleton
    stop_proxy()

    _auth_proxy_singleton = AuthProxy(
        target_addr=target_addr,
        auth_token=auth_token,
        listen_host=listen_host,
        listen_port=listen_port,
    )
    _auth_proxy_singleton.start()
    return _auth_proxy_singleton


def get_proxy() -> AuthProxy | None:
    """Get the current auth proxy singleton instance."""
    return _auth_proxy_singleton


@atexit.register
def stop_proxy():
    global _auth_proxy_singleton
    if _auth_proxy_singleton:
        _auth_proxy_singleton.stop()
        _auth_proxy_singleton = None
