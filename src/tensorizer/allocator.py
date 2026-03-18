import bisect
import mmap
import os
import struct
import weakref
from typing import Final

from tensorizer.frame import DType, MemFrame, MemLoc

ALLOC_CAPACITY: Final[int] = int(os.environ.get("ALLOC_CAPACITY", str(10 << 10)))


class _WeakSingleton(type):
    __instance: weakref.ReferenceType[Allocator] | None = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is not None:
            instance = cls.__instance()
            if instance is not None:
                return instance

        instance = super().__call__(*args, **kwargs)
        instance._open(ALLOC_CAPACITY)
        cls.__instance = weakref.ref(instance)
        return instance


class Allocator(metaclass=_WeakSingleton):
    def __init__(self) -> None:
        self.__mm = None
        self.__capacity = None
        self.__allocs = weakref.WeakKeyDictionary()
        self.__offset = 0
        self.__freelocs = []

    def _open(self, capacity: int) -> None:
        if self.__mm is not None:
            raise TypeError

        self.__mm = mmap.mmap(-1, capacity)
        self.__capacity = capacity

        weakref.finalize(self, mmap.mmap.close, self.__mm)

    def alloc(self, tensor: "Tensor", count: int, dtype: DType) -> MemFrame:
        allocsize = count * struct.calcsize(dtype)
        memloc = self.__retrieve_memloc(allocsize)
        memframe = MemFrame(
            self.__mm, memloc, tt=dtype, notify_release=self._on_release
        )
        self.__allocs[tensor] = memframe
        return memframe

    def collect(self) -> None:
        """
        Check median & total __freelocs
        Use dontneed
        """

    def _on_release(self, memloc: MemLoc) -> None:
        bisect.insort_right(self.__freelocs, memloc, key=lambda loc: loc.allocsize)

    def __retrieve_memloc(self, allocsize: int) -> MemLoc:
        pos = bisect.bisect_left(
            self.__freelocs, allocsize, key=lambda loc: loc.allocsize
        )
        if pos < len(self.__freelocs):
            return self.__freelocs.pop(pos)

        aligned = self.__paginate(self.__offset)
        if aligned + allocsize > self.__capacity:
            raise MemoryError

        memloc = MemLoc(start=aligned, allocsize=allocsize)
        self.__offset = memloc.stop
        return memloc

    @staticmethod
    def __paginate(loc: int) -> int:
        return (loc + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
