import copy
import threading
from typing import TypeVar, Generic

T = TypeVar("T")


class LatestResult(Generic[T]):
    """Thread-safe single-slot store. Writers always overwrite; readers get the most recent value."""

    def __init__(self, initial: T):
        self._lock = threading.Lock()
        self._value: T = initial

    def put(self, value: T) -> None:
        with self._lock:
            self._value = copy.deepcopy(value)

    def get(self) -> T:
        with self._lock:
            return copy.deepcopy(self._value)
