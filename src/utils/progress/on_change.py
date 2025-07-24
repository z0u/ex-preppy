from typing import Any, Generic, Self, SupportsIndex, TypeVar, overload, Protocol, override


T = TypeVar('T')
_VT = TypeVar('_VT')


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

    @override
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._on_change()

    @override
    def __delitem__(self, key):
        super().__delitem__(key)
        self._on_change()

    @override
    def __ior__(self, value, /) -> Self:
        super().__ior__(value)
        self._on_change()
        return self

    @override
    def clear(self):
        super().clear()
        self._on_change()

    @override
    def pop(self, *args) -> V:
        value = super().pop(*args)
        self._on_change()
        return value

    @override
    def popitem(self) -> tuple[K, V]:
        value = super().popitem()
        self._on_change()
        return value

    @overload
    def setdefault(self, key: K, default: None = ...) -> V | None: ...
    @overload
    def setdefault(self, key: K, default: _VT) -> V | _VT: ...
    @override
    def setdefault(self, key, default: Any = None) -> Any:
        key_existed = key in self
        value = super().setdefault(key, default)
        if not key_existed:
            self._on_change()
        return value


class OnChangeDictProp[K, V](OnChangeProp[dict[K, V]]):
    def __set__(self, instance: ChangeObserver, value: dict[K, V]) -> None:
        store = OnChangeDict[K, V](value)
        store._on_change = instance._on_change
        super().__set__(instance, value)


class OnChangeList(list[T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_change = lambda: None

    @override
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._on_change()

    @override
    def __delitem__(self, key):
        super().__delitem__(key)
        self._on_change()

    @override
    def __iadd__(self, value) -> Self:
        self.extend(value)
        return self

    @override
    def append(self, item):
        super().append(item)
        self._on_change()

    @override
    def extend(self, iterable):
        super().extend(iterable)
        self._on_change()

    @override
    def insert(self, index, item):
        super().insert(index, item)
        self._on_change()

    @override
    def pop(self, index: SupportsIndex = -1) -> T:
        item = super().pop(index)
        self._on_change()
        return item

    @override
    def remove(self, item):
        super().remove(item)
        self._on_change()

    @override
    def clear(self):
        super().clear()
        self._on_change()


class OnChangeListProp(OnChangeProp[list[T]]):
    def __set__(self, instance: ChangeObserver, value: list[T]) -> None:
        store = OnChangeList[T](value)
        store._on_change = instance._on_change
        super().__set__(instance, value)
