import logging
from typing import override

from aim.pytorch_lightning import AimLogger
from aim.sdk.run import Run
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities import rank_zero_only

from track.patches.client import patch_aim_client

log = logging.getLogger(__name__)


class AuthAimLogger(AimLogger):
    """[Aim](https://github.com/aimhubio/aim) logger with Basic auth."""

    def __init__(self, *args, api_key: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key

    @property
    @rank_zero_experiment
    @override
    def experiment(self) -> Run:
        patch_aim_client(self.api_key)
        return super().experiment

    @rank_zero_only
    def finalize(self, status: str = ''):
        log.info(f'Finalizing logger with status {status}...')
        super().finalize(status)
        log.info('Finalized.')
