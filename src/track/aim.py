import logging
import os
from pathlib import PurePosixPath
from typing import cast

import modal
from yarl import URL

import infra
from infra.requirements import uv_freeze, modnames
from track.patches.client import patch_aim_client

AIM_DIR = PurePosixPath('/aim')
THIRD_PARTY_PACKAGES = uv_freeze(groups=['track'], indexes=['https://download.pytorch.org/whl/cpu'])

log = logging.getLogger(__name__)

track_app_name = 'mi-ni.track'


image = (
    modal.Image.debian_slim()  #
    .pip_install(*THIRD_PARTY_PACKAGES, extra_index_url='https://download.pytorch.org/whl/cpu')
    .add_local_python_source(*modnames('self', infra))
)
volume = modal.Volume.from_name(track_app_name, create_if_missing=True)
app = modal.App(
    track_app_name,
    image=image,
    volumes={AIM_DIR: volume} if volume else {},
)


@app.cls(
    secrets=[modal.Secret.from_name('google-oauth')],
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
class AimService:
    internal_api_key: str | None

    @modal.asgi_app(label='track')
    def web_interface(self):
        """Start the web app and its services."""
        from track.auth_wrapper import create_wrapper_app

        self._ensure_repo(AIM_DIR)
        os.chdir(AIM_DIR)

        # Aim itself doesn't support authentication (at least, not in the community edition)
        # Wrap the server with middleware to authenticate users and services
        # Services will need to provide the API key when making requests.
        app, self.internal_api_key = create_wrapper_app(track_app_name)

        from aim.ext.transport.server import create_app as create_aim_services
        from aim.web.api import create_app as create_aim_web_app

        app.mount('/server', create_aim_services(), 'Aim services')
        app.mount('/', create_aim_web_app(), 'Aim')
        return app

    @modal.method()
    def get_repo_for_client(self):
        """Get a factory to create Repo objects that connect to this remote service."""
        url = self.web_interface.get_web_url()  # Created by Modal
        if not self.internal_api_key or not url:
            raise RuntimeError("Server isn't running")
        return url, self.internal_api_key

    def _ensure_repo(self, path: str | PurePosixPath) -> None:
        from aim.sdk.repo import Repo

        if not Repo.exists(str(AIM_DIR)):
            log.info(f'Creating Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR), init=True)
        else:
            log.info(f'Using existing Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR))


async def get_repo():
    from aim.sdk.repo import Repo

    aim_service = modal.Cls.from_name(track_app_name, 'AimService')()
    url, internal_api_key = await aim_service.get_repo_for_client.remote.aio()
    url = cast(str, url)

    # Patch the Aim Client class to provide the API key.
    patch_aim_client(internal_api_key)

    aim_url = URL(url).with_scheme('aim') / 'server'
    return Repo(str(aim_url))
