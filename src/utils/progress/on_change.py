from typing import Generic, TypeVar, overload, Protocol


T = TypeVar('T')


class ChangeObserver(Protocol):
    def _on_change(self): ...


class OnChangeProp(Generic[T]):
    """A descriptor that triggers a display update on set."""

    private_name: str

    def __set_name__(self, owner: type[ChangeObserver], name: str):
        self.private_name = f'_{name}'

    @overload
    def __get__(self, instance: None, owner: type[ChangeObserver]) -> 'OnChangeProp[T]': ...

    @overload
    def __get__(self, instance: ChangeObserver, owner: type[ChangeObserver]) -> T: ...

    def __get__(self, instance: ChangeObserver | None, owner: type[ChangeObserver]) -> T | 'OnChangeProp[T]':
        if instance is None:
            return self
        return getattr(instance, self.private_name)

    def __set__(self, instance: ChangeObserver, value: T) -> None:
        setattr(instance, self.private_name, value)
        instance._on_change()


class OnChangeDict[K, V](dict[K, V]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_change = lambda: None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._on_change()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._on_change()


class OnChangeDictProp[K, V](OnChangeProp[dict[K, V]]):
    def __set__(self, instance: ChangeObserver, value: dict[K, V]) -> None:
        store = OnChangeDict[K, V](value)
        store._on_change = instance._on_change
        super().__set__(instance, value)
