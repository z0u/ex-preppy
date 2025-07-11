import secrets
import modal


def get_or_create_session_secret(app_name: str) -> str:
    """Get or create a session secret, storing it persistently in Modal."""
    secrets_dict = modal.Dict.from_name('dynamic-secrets', create_if_missing=True)
    key = f'{app_name}-session_secret'

    if key in secrets_dict:
        return secrets_dict[key]

    new_secret = secrets.token_urlsafe(nbytes=256 // 8)
    secrets_dict[key] = new_secret

    return new_secret
