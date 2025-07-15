from typing import Any
from utils.progress.on_change import OnChangeDictProp, OnChangeProp


class _Progress:
    total: OnChangeProp[int] = OnChangeProp[int]()
    """
    The total number of items.

    Changing this will schedule a redraw.
    """

    count: OnChangeProp[int] = OnChangeProp[int]()
    """
    The number of items that have been yielded or completed.

    Changing this will schedule a redraw.
    """

    description: OnChangeProp[str] = OnChangeProp[str]()
    """
    A description to display before the stats.

    Changing this will schedule a redraw.
    """

    suffix: OnChangeProp[str] = OnChangeProp[str]()
    """
    Additional information to display after the stats.

    Changing this will schedule a redraw.
    """

    metrics: OnChangeDictProp[str, Any] = OnChangeDictProp[str, Any]()
    """
    Metrics to display under the bar.

    Changing this will schedule a redraw.
    """

    def _on_change(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
