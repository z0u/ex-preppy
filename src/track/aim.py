from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import modal
from yarl import URL

import infra
from infra.requirements import modnames, uv_freeze
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
    # Limit to one container. In this simple setup, the database is on a Modal Volume,
    # which only allows IPC via files within the one container. Multiple containers
    # _could_ write to it, but that would require commit/refresh calls which are slow
    # and may fail if the same file is written to by multiple workers.
    max_containers=1,
    scaledown_window=60,
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
    def get_repo_for_client(self) -> RepoLoc:
        """Get a factory to create Repo objects that connect to this remote service."""
        url = self.web_interface.get_web_url()  # Created by Modal
        if not self.internal_api_key or not url:
            raise RuntimeError("Server isn't running")
        return RepoLoc(
            url=str(url),
            aim_url=str(URL(url).with_scheme('aim') / 'server'),
            api_key=self.internal_api_key,
        )

    def _ensure_repo(self, path: str | PurePosixPath) -> None:
        from aim.sdk.repo import Repo

        if not Repo.exists(str(AIM_DIR)):
            log.info(f'Creating Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR), init=True)
        else:
            log.info(f'Using existing Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR))


@dataclass
class RepoLoc:
    url: str
    aim_url: str
    api_key: str = field(repr=False)


def get_repo_loc():
    aim_service = modal.Cls.from_name(track_app_name, 'AimService')()
    repo_loc = aim_service.get_repo_for_client.remote()
    assert isinstance(repo_loc, RepoLoc)
    return repo_loc


def get_repo():
    from aim.sdk.repo import Repo

    repo_loc = get_repo_loc()

    # Patch the Aim Client class to provide the API key.
    patch_aim_client(repo_loc.api_key)

    return Repo(str(repo_loc.aim_url))
