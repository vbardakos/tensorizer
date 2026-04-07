import math

from simd import add_f32, mul_f32, neg_f32, sum_f32

from tensorizer.allocator import Allocator


def add(a, b):
    b = _coerce(a, b)
    _check_shapes(a, b)
    return _binop(a, b, add_f32)


def mul(a, b):
    b = _coerce(a, b)
    _check_shapes(a, b)
    return _binop(a, b, mul_f32)


def neg(a):
    result = neg_f32(_buffer(a))
    return _result_tensor(a, result)


def sub(a, b):
    return add(a, neg(b))


def dot(a, b):
    if len(a.shape) != 1 or len(b.shape) != 1:
        msg = "Dot product requires 1D tensors"
        raise ValueError(msg)
    _check_shapes(a, b)
    products = mul_f32(_buffer(a), _buffer(b))
    tmp = _result_tensor(a, products)
    return sum_f32(_buffer(tmp))


def reduce_sum(a):
    return sum_f32(_buffer(a))


# =============================
# ========= INTERNALS =========
# =============================


def _coerce(a, other):
    from tensorizer.tensor import Tensor

    if isinstance(other, Tensor):
        return other
    if isinstance(other, (int, float)):
        return Tensor.fill(shape=a.shape, dtype=a.dtype, value=other)
    return NotImplemented


def _check_shapes(a, b):
    if b is NotImplemented:
        return
    if a.shape != b.shape:
        msg = f"Shape mismatch: {a.shape} vs {b.shape}"
        raise ValueError(msg)


def _binop(a, b, op):
    if b is NotImplemented:
        return NotImplemented
    result = op(_buffer(a), _buffer(b))
    return _result_tensor(a, result)


def _buffer(tensor):
    from tensorizer.tensor import Tensor

    if tensor.is_contiguous:
        return tensor._frame.buffer()

    data = [tensor[idx] for idx in _iter_indices(tensor.shape)]

    materialized = Tensor(data, shape=tensor.shape, dtype=tensor.dtype)
    return materialized._frame.buffer()


def _result_tensor(source, simd_result):
    from tensorizer.tensor import Tensor

    count = math.prod(source.shape)
    allocator = Allocator()
    frame = allocator.alloc_raw(count, source.dtype)
    frame.write_bytes(bytes(simd_result))
    return Tensor._from_frame(frame, shape=source.shape, dtype=source.dtype)


def _iter_indices(shape):
    if len(shape) == 1:
        yield from range(shape[0])
    else:
        for i in range(shape[0]):
            for rest in _iter_indices(shape[1:]):
                yield (i, *rest) if isinstance(rest, tuple) else (i, rest)
