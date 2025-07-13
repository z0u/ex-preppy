from functools import lru_cache
import secrets
from typing import Any, Callable
import modal


def get_session_secret(app_name: str) -> str:
    """Get or create a session secret, storing it persistently in Modal."""
    store = _get_store(app_name)
    return _get_or_create(store, 'session_secret', _strong_secret_factory)


def get_internal_api_key(app_name: str) -> str:
    """Get or create a session secret, storing it persistently in Modal."""
    store = _get_store(app_name)
    return _get_or_create(store, 'internal_api_key', _strong_secret_factory)


def _strong_secret_factory():
    return secrets.token_urlsafe(nbytes=256 // 8)


def _get_or_create(store: modal.Dict, key: str, factory: Callable[[], Any]):
    if key not in store:
        # Skip if exists, in case another worker added a value
        store.put(key, factory(), skip_if_exists=True)
    return store[key]


@lru_cache
def _get_store(app_name: str) -> modal.Dict:
    return modal.Dict.from_name(f'{app_name}-dynamic-secrets', create_if_missing=True)
