import mmap
import struct
import weakref
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum


class DType(StrEnum):
    I32 = "i"
    F32 = "f"


@dataclass(slots=True, frozen=True)
class MemLoc:
    start: int
    allocsize: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def stop(self) -> int:
        return self.start + self.allocsize


class MemFrame:
    def __init__(
        self,
        mm: mmap.mmap,
        memloc: MemLoc,
        tt: DType,
        notify_release: Callable[[MemLoc], None],
        relloc: MemLoc | None = None,
        refs: weakref.WeakSet | None = None,
    ) -> None:
        if not refs:
            self.__mm.madvise(mmap.MADV_WILLNEED, memloc.start, memloc.allocsize)

        self.__mm = mm
        self.__memloc = memloc
        self.__relloc = relloc or memloc

        self.__dtype = tt
        self.__unitsize = struct.calcsize(tt)

        self.__on_release = notify_release

        refs = refs or weakref.WeakSet()
        refs.add(self)
        self.__refs = refs

        weakref.finalize(self, MemFrame._release, mm, memloc, refs, notify_release)

    @staticmethod
    def _release(
        mm: mmap.mmap,
        memloc: MemLoc,
        refs: weakref.WeakSet,
        on_release: Callable[[MemLoc], None],
    ) -> None:
        if any(ref for ref in refs):
            return

        with suppress(Exception):
            mm.madvise(mmap.MADV_FREE, memloc.start, memloc.allocsize)
        on_release(memloc)

    def raw_ptr(self, reloffset: int, allocsize: int) -> MemFrame:
        if not (reloffset or allocsize):
            return self._new_ptr(None)

        relloc = MemLoc(self.__relloc.start + reloffset, allocsize)
        if self.memoffset > relloc.start or self.__memloc.stop < relloc.stop:
            msg = "Pointer exceeds frame's allocated space"
            raise MemoryError(msg)
        return self._new_ptr(relloc)

    def ptr(self, from_idx: int, size: int) -> MemFrame:
        if not (from_idx or size):
            return self._new_ptr(None)

        reloffset = from_idx * self.__unitsize
        allocsize = size * self.__unitsize

        relloc = MemLoc(start=self.__relloc.start + reloffset, allocsize=allocsize)
        if self.memoffset > relloc.start or self.__memloc.stop < relloc.stop:
            msg = "Pointer exceeds frame's allocated space"
            raise MemoryError(msg)

        return self._new_ptr(relloc)

    def _new_ptr(self, relloc: MemLoc | None) -> MemFrame:
        return MemFrame(
            mm=self.__mm,
            memloc=self.__memloc,
            relloc=relloc,
            notify_release=self.__on_release,
            tt=self.__dtype,
            refs=self.__refs,
        )

    def write(self, data: bytes) -> None:
        self.__mm[self.__relloc.slice] = data

    def fill(self, value) -> None:
        packed = struct.pack(self.__dtype, value)
        self.__mm[self.__relloc.slice] = packed * len(self)

    def raw_buf(self) -> memoryview:
        return memoryview(self.__mm)[self.__relloc.slice]

    def buf(self) -> memoryview:
        return self.raw_buf().cast(self.__dtype)

    def is_referenced(self) -> bool:
        return len(self.__refs) > 1

    @property
    def memoffset(self) -> int:
        return self.__memloc.start

    @property
    def allocsize(self) -> int:
        return self.__memloc.allocsize

    def __len__(self) -> int:
        return self.__relloc.allocsize // self.__unitsize

    def __repr__(self) -> str:
        return f"MemoryFrame(memoffset={self.memoffset}, allocsize={self.allocsize}, size={len(self)})"  # noqa: E501
