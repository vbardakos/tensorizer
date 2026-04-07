import math

from tensorizer import _ops
from tensorizer.allocator import Allocator
from tensorizer.frame import DType


class Tensor:
    __slots__ = (
        "__dtype",
        "__frame",
        "__offset",
        "__shape",
        "__strides",
        "__weakref__",
    )

    def __init__(self, data, *, shape, dtype=DType.F32):
        shape = tuple(shape)
        count = math.prod(shape)
        if len(data) != count:
            msg = f"Data length {len(data)} != shape {shape} (expected {count})"
            raise ValueError(msg)

        allocator = Allocator()
        self.__frame = allocator.alloc(self, count, dtype)
        self.__frame.write(data)
        self.__shape = shape
        self.__strides = _strides_for(shape)
        self.__dtype = dtype
        self.__offset = 0

    @classmethod
    def zeros(cls, *, shape, dtype=DType.F32):
        return cls.fill(shape=shape, dtype=dtype, value=0)

    @classmethod
    def fill(cls, *, shape, dtype=DType.F32, value):
        shape = tuple(shape)
        count = math.prod(shape)
        obj = object.__new__(cls)
        allocator = Allocator()
        obj.__frame = allocator.alloc(obj, count, dtype)
        obj.__frame.fill(value)
        obj.__shape = shape
        obj.__strides = _strides_for(shape)
        obj.__dtype = dtype
        obj.__offset = 0
        return obj

    @classmethod
    def _from_frame(cls, frame, *, shape, dtype):
        obj = object.__new__(cls)
        obj.__frame = frame
        obj.__shape = tuple(shape)
        obj.__strides = _strides_for(shape)
        obj.__dtype = dtype
        obj.__offset = 0
        return obj

    @classmethod
    def _view(cls, frame, shape, strides, dtype, offset):
        obj = object.__new__(cls)
        obj.__frame = frame
        obj.__shape = shape
        obj.__strides = strides
        obj.__dtype = dtype
        obj.__offset = offset
        return obj

    def transpose(self):
        if len(self.__shape) < 2:
            return self
        shape = (*self.__shape[:-2], self.__shape[-1], self.__shape[-2])
        strides = (*self.__strides[:-2], self.__strides[-1], self.__strides[-2])
        return Tensor._view(self.__frame, shape, strides, self.__dtype, self.__offset)

    def reshape(self, *shape):
        if math.prod(shape) != len(self):
            msg = f"Cannot reshape {self.__shape} into {shape}"
            raise ValueError(msg)
        if not self.is_contiguous:
            msg = "Cannot reshape non-contiguous tensor"
            raise ValueError(msg)
        return Tensor._view(
            self.__frame, shape, _strides_for(shape), self.__dtype, self.__offset
        )

    def __add__(self, other):
        return _ops.add(self, other)

    def __mul__(self, other):
        return _ops.mul(self, other)

    def __neg__(self):
        return _ops.neg(self)

    def __sub__(self, other):
        return _ops.sub(self, other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return _ops.dot(self, other)

    def __radd__(self, other):
        return _ops.add(self, other)

    def __rmul__(self, other):
        return _ops.mul(self, other)

    def __rsub__(self, other):
        return _ops.add(_ops.neg(self), other)

    def sum(self):
        return _ops.reduce_sum(self)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = (idx,)

        offset = self.__offset
        for i, s in zip(idx, self.__strides):
            offset += i * s

        remaining = self.__shape[len(idx) :]
        if not remaining:
            return self.__frame.buffer()[offset]

        return Tensor._view(
            self.__frame, remaining, self.__strides[len(idx) :], self.__dtype, offset
        )

    @property
    def _frame(self):
        return self.__frame

    @property
    def shape(self):
        return self.__shape

    @property
    def strides(self):
        return self.__strides

    @property
    def dtype(self):
        return self.__dtype

    @property
    def is_contiguous(self):
        return self.__strides == _strides_for(self.__shape)

    def __len__(self):
        return math.prod(self.__shape)

    def __repr__(self):
        return f"Tensor(shape={self.__shape}, dtype={self.__dtype.name})"


def _strides_for(shape):
    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))
