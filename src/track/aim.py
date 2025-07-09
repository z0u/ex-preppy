import logging
import os
from pathlib import PurePosixPath

import modal

from infra.requirements import freeze, get_project_name

assert __package__, 'This module must be run as a package.'

AIM_DIR = PurePosixPath('/aim')
LOCAL_PACKAGES = (__package__, 'infra')


log = logging.getLogger(__name__)


def create_app(name: str | None = None):
    name = f'{get_project_name()}.{__package__}'
    image = (
        modal.Image.debian_slim()  #
        .pip_install(*freeze(group='track'))
        .workdir('/tmp')
        .add_local_file('pyproject.toml', '/tmp/pyproject.toml')
        .add_local_python_source(*LOCAL_PACKAGES)
    )
    volume = modal.Volume.from_name(name, create_if_missing=True)
    app = modal.App(
        name,
        image=image,
        volumes={AIM_DIR: volume},
    )

    @app.function(serialized=True)
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app()
    def start_aim():
        from aim.sdk.repo import Repo

        if not Repo.exists(str(AIM_DIR)):
            log.info(f'Creating Aim repo at {AIM_DIR}')
            Repo.from_path(str(AIM_DIR), init=True)
        else:
            log.info(f'Aim repo exists at {AIM_DIR}')

        os.chdir(AIM_DIR)

        from aim.web.run import app as aim_asgi_app

        return aim_asgi_app

    return app


if modal.is_local:
    app = create_app()
