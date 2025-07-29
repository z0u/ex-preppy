from utils.progress.progress import ProgressBase


class DummyProgress(ProgressBase):
    def __init__(self):
        super().__init__(total=1, description='', initial_metrics={})
        self.calls = []

    def before_print(self):
        self.calls.append('before')

    def after_print(self):
        self.calls.append('after')

    def _debounced_draw(self):
        pass
