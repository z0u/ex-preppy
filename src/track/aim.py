import logging
import os
import typing
from pathlib import PurePosixPath

import modal
from aim.sdk.repo import Repo

from infra.requirements import freeze
from track.repo_factory import ProxyRepoFactory

assert __package__, 'This module must be run as a package.'

AIM_DIR = PurePosixPath('/aim')
LOCAL_PACKAGES = (__package__, 'infra')
THIRD_PARTY_PACKAGES = freeze(group='track') if modal.is_local() else []


log = logging.getLogger(__name__)

track_app_name = 'mi-ni.track'


image = (
    modal.Image.debian_slim()  #
    .pip_install(*THIRD_PARTY_PACKAGES)
    .add_local_python_source(*LOCAL_PACKAGES)
)
volume = modal.Volume.from_name(track_app_name, create_if_missing=True)
app = modal.App(
    track_app_name,
    image=image,
    volumes={AIM_DIR: volume} if volume else {},
)


@app.cls(
    # serialized=True,  # Isolate from the rest of this module
    secrets=[modal.Secret.from_name('google-oauth')],
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
class AimService:
    internal_api_key: str | None

    @modal.asgi_app(label='track')
    def web_interface(self):
        """Start the web app and its services."""
        # Change to the repo directory because aim.web.run will create the app on import
        self._ensure_repo(AIM_DIR)
        os.chdir(AIM_DIR)

        from aim.web.run import app as aim_asgi_app

        from track.auth_wrapper import create_wrapper_app

        # Aim itself doesn't support authentication (at least, not in the community edition)
        # Wrap the server with middleware to authenticate users and services
        # A proxy server will need to be used to connect from services (see repo_factory)
        wrapper_app, self.internal_api_key = create_wrapper_app(track_app_name)
        wrapper_app.mount('/', aim_asgi_app, 'Aim')
        return wrapper_app

    @modal.method()
    def get_repo_for_client(self) -> typing.Callable[[], Repo]:
        """Get a factory to create Repo objects that connect to this remote service."""
        target_url = self.web_interface.web_url  # Created by Modal
        if not self.internal_api_key or not target_url:
            raise RuntimeError("Server isn't running")
        return ProxyRepoFactory(target_url, self.internal_api_key)

    def _ensure_repo(self, path: str | PurePosixPath) -> None:
        if not Repo.exists(str(AIM_DIR)):
            log.info(f'Creating Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR), init=True)
        else:
            log.info(f'Using existing Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR))


async def get_repo():
    cls = modal.Cls.from_name(track_app_name, 'AimService')
    aim_service = cls()
    repo_factory = typing.cast(ProxyRepoFactory, await aim_service.get_repo_for_client.remote.aio())
    repo = repo_factory()
    return repo
