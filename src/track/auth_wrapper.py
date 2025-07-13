import os
from typing import Awaitable, Callable, TypeAlias

from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from track.secrets import get_session_secret, get_internal_api_key

MiddlewareDelegate: TypeAlias = Callable[[Request], Awaitable[Response]]


def create_wrapper_app(name: str):  # noqa: C901
    """
    Create a FastAPI app to wrap another one with an authentication flow.

    Usage:

    ```python
    from track.auth_wrapper import create_wrapper_app
    app = create_wrapper_app('my_app_name')
    other_app = ...  # Your actual app, e.g., Aim's app
    app.mount('/', other_app)
    ```
    """
    app = FastAPI(name=name)
    """Wrapper app that provides authentication middleware"""

    internal_api_key = get_internal_api_key(name)
    """Secret token for access by other Modal workers"""

    # OAuth client config (from Google Cloud Console)
    oauth = OAuth()
    oauth.register(
        name='google',
        client_id=os.environ['GOOGLE_CLIENT_ID'],
        client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email',
            'response_type': 'code',
        },
    )

    def is_authorized(request: Request) -> bool:
        return _is_authorized_service(request) or _is_authorized_user(request)

    def _is_authorized_service(request: Request) -> bool:
        # Service accounts use an API key
        authz = request.headers.get('Authorization', '')
        if authz == f'Bearer {internal_api_key}':
            return True
        return False

    def _is_authorized_user(request: Request) -> bool:
        # Users use OAuth
        user_email = request.session.get('user', {}).get('email')
        if not user_email:
            return False

        allowed_emails_str = os.environ.get('ALLOWED_EMAIL', '')
        if not allowed_emails_str:
            return False

        allowed_emails = {email.strip() for email in allowed_emails_str.split(',')}
        return user_email in allowed_emails

    @app.middleware('http')
    async def auth_middleware(request: Request, call_next: MiddlewareDelegate):
        # Ensures all requests are authorized, except for login/auth endpoints
        login_path = request.url_for('login').path
        auth_path = request.url_for('auth').path
        if request.url.path in (login_path, auth_path):
            return await call_next(request)

        if not is_authorized(request):
            if not os.environ.get('ALLOWED_EMAIL'):
                return JSONResponse({'error': 'authentication is misconfigured'}, status_code=500)
            return RedirectResponse(request.url_for('login'))
        return await call_next(request)

    # This needs to be added after our @app.middleware function, or the session won't be available.
    app.add_middleware(SessionMiddleware, secret_key=get_session_secret(name))

    @app.get('/login')
    async def login(request: Request):
        # Redirect to Google for login
        redirect_uri = request.url_for('auth')
        return await oauth.google.authorize_redirect(request, redirect_uri)  # type: ignore

    @app.get('/auth')
    async def auth(request: Request):
        # This is the callback URL that Google redirects to after login
        # The request contains the authorization code; exchange it for an access token
        token: dict = await oauth.google.authorize_access_token(request)  # type: ignore
        user = token['userinfo']
        request.session['user'] = dict(user)

        # Re-use the same authorization check as the middleware
        if not is_authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=403)

        return RedirectResponse('/')

    return app, internal_api_key
