import bisect
import mmap
import os
import struct
import weakref
from typing import TYPE_CHECKING, Final

from tensorizer.frame import DType, MemoryFrame, Span

if TYPE_CHECKING:
    from tensorizer.tensor import Tensor

ARENA_CAPACITY: Final[int] = int(os.environ.get("ARENA_CAPACITY", str(10 << 10)))


class _EphemeralSingleton(type):
    __instance: weakref.ReferenceType[Allocator] | None = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is not None:
            instance = cls.__instance()
            if instance is not None:
                return instance

        instance: Allocator = super().__call__(*args, **kwargs)
        instance._setup(ARENA_CAPACITY)
        cls.__instance = weakref.ref(instance)
        return instance


class Allocator(metaclass=_EphemeralSingleton):
    def __init__(self) -> None:
        self.__mm = None
        self.__capacity = None
        self.__allocs = weakref.WeakKeyDictionary()
        self.__offset = 0
        self.__freelocs = []

    def _setup(self, capacity: int) -> None:
        if self.__mm is not None:
            raise TypeError

        self.__mm = mmap.mmap(-1, capacity)
        self.__capacity = capacity

        # unsafe: all mmap ptrs need cleanup first
        weakref.finalize(self, mmap.mmap.close, self.__mm)

    def alloc(self, tensor: Tensor, count: int, dtype: DType) -> MemoryFrame:
        """
        indices
        ---
        span:       offset :: aligned(offset + size)
        subspan:    offset :: offset + size
        remainder:  offset + size :: aligned(offset + size)

        check freelocs -> assign to freelocs
        create new -> ...
        """
        frame = self.alloc_raw(count, dtype)
        self.__allocs[tensor] = frame
        return frame

    def alloc_raw(self, count: int, dtype: DType) -> MemoryFrame:
        size = count * struct.calcsize(dtype)
        span = self.__create_span(size)
        subspan = Span(span.offset, size) if span.size != size else None
        memframe = MemoryFrame(
            self.__mm,
            span,
            dtype=dtype,
            on_dealloc=self._on_dealloc,
            subspan=subspan,
        )
        self.__stash_remainder(span, dtype)
        return memframe

    def collect(self) -> None:
        """
        use freeloc stats (eg Check median & total freelocs)
        use madv dontneed (frame uses lazy free)
        """

    def _on_dealloc(self, span: Span) -> None:
        """Retrieves free frame for reallocation"""
        bisect.insort_right(self.__freelocs, span, key=lambda span: span.size)

    def __create_span(self, size: int) -> Span:
        # Check if there's a existing free span which fits the new alloc
        idx = bisect.bisect_left(self.__freelocs, size, key=lambda span: span.size)
        if idx < len(self.__freelocs):
            return self.__freelocs.pop(idx)

        aligned = self.__align(self.__offset)
        if aligned + size > self.__capacity:
            raise MemoryError

        span = Span(offset=aligned, size=size)
        self.__offset = span.end
        return span

    def __stash_remainder(self, span: Span, dtype: DType) -> None:
        remainder_size = self.__align(span.end) - span.end

        if remainder_size < struct.calcsize(dtype) * 4:
            return

        remainder = Span(span.end, remainder_size)
        self._on_dealloc(remainder)

    @staticmethod
    def __align(offset: int) -> int:
        return (offset + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
