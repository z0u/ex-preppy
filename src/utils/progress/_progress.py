from typing import Any
from utils.progress.on_change import OnChangeDictProp, OnChangeProp


class _Progress:
    total = OnChangeProp[int]()
    count = OnChangeProp[int]()
    description = OnChangeProp[str]()
    suffix = OnChangeProp[str]()
    metrics = OnChangeDictProp[str, Any]()

    def _on_change(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
