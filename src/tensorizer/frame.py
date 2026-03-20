import itertools
import mmap
import struct
import sys
import weakref
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class DType(StrEnum):
    I32 = "i"
    F32 = "f"


@dataclass(slots=True, weakref_slot=True, frozen=True)
class Span:
    offset: int
    size: int

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.end)

    @property
    def end(self) -> int:
        return self.offset + self.size

    def refcount(self) -> int:
        """Counter of strong references."""
        return sys.getrefcount(self) - 1


class MemoryFrame:
    def __init__(
        self,
        mm: mmap.mmap,
        span: Span,
        dtype: DType,
        on_dealloc: Callable[[Span], None],
        subspan: Span | None = None,
        contiguous: bool = True,
    ) -> None:
        self.contiguous = True

        self.__mm = mm
        self.__span = span
        self.__subspan = subspan or Span(span.offset, span.size)
        self.__dtype = dtype
        self.__unitsize = struct.calcsize(dtype)
        self.__contiguous = contiguous
        self.__on_dealloc = on_dealloc

        if subspan is None:
            self.__mm.madvise(mmap.MADV_WILLNEED, span.offset, span.size)

            weakref.finalize(
                span, MemoryFrame._dealloc, mm, span.offset, span.size, on_dealloc
            )

    @staticmethod
    def _dealloc(
        mm: mmap.mmap,
        offset: int,
        size: int,
        on_dealloc: Callable[[Span], None],
    ) -> None:
        with suppress(Exception):
            mm.madvise(mmap.MADV_FREE, offset, size)
        on_dealloc(Span(offset, size))

    def ptr(self, idx: int, count: int, contiguous: bool) -> MemoryFrame:
        if not idx and count == len(self):
            return self._new_ptr(self.__subspan, contiguous)

        offset = self.__subspan.offset + idx * self.__unitsize
        size = count * self.__unitsize

        subspan = Span(offset=offset, size=size)
        if self.offset > subspan.offset or self.__span.end < subspan.end:
            msg = "Pointer exceeds frame's allocated space"
            raise MemoryError(msg)

        return self._new_ptr(subspan, contiguous)

    def _new_ptr(self, subspan: Span, contiguous: bool) -> MemoryFrame:
        return MemoryFrame(
            mm=self.__mm,
            span=self.__span,
            subspan=subspan,
            on_dealloc=self.__on_dealloc,
            dtype=self.__dtype,
            contiguous=contiguous,
        )

    def write(self, data: Sequence) -> None:
        if self.shares_ownership():
            msg = "Cannot write on shared Memory space"
            raise BufferError(msg)
        fmt = f"{len(data)}{self.__dtype}"
        self.__mm[self.__subspan.slice] = struct.pack(fmt, *data)

    def write_bytes(self, data: bytes) -> None:
        if self.shares_ownership():
            msg = "Cannot write on shared Memory space"
            raise BufferError(msg)
        self.__mm[self.__subspan.slice] = data

    def fill(self, value: Any) -> None:
        if self.shares_ownership():
            msg = "Cannot write on shared Memory space"
            raise BufferError(msg)

        fmt = f"{len(self)}{self.__dtype}"
        values = itertools.repeat(value, len(self))
        struct.pack_into(fmt, self.__mm, self.__subspan.offset, *values)

    def raw_buf(self) -> memoryview:
        return memoryview(self.__mm)[self.__subspan.slice]

    def buf(self) -> memoryview:
        return self.raw_buf().cast(self.__dtype)

    def shares_ownership(self) -> bool:
        return self.__span.refcount() > 1

    @property
    def offset(self) -> int:
        return self.__span.offset

    @property
    def size(self) -> int:
        return self.__span.size

    @property
    def dtype(self) -> DType:
        return self.__dtype

    def __len__(self) -> int:
        return self.__subspan.size // self.__unitsize

    def __repr__(self) -> str:
        return f"MemoryFrame(offset={self.offset}, size={self.size}, shared={self.shares_ownership()})"  # noqa: E501
