# Tensorizer

An honest attempt at data-oriented design in Python.
Memory layout drives the architecture — flat buffers, arena allocation, stride-based views.
The OOP surface exists because a library needs an interface, not because the design called for it.

No numpy, no external libraries — just Python's builtins and a bit of Rust for SIMD.
There's no magic here. Everything the engine does — memory mapping, page hints, reference-counted lifetimes — is built on top of what the kernel and the Python runtime already provide.
The pieces were always there.

## Ground rules

This project sets a few constraints on itself to keep things honest:

- **No external Python libraries.** Everything is built on Python's builtins. If the runtime doesn't ship it, we don't use it.
- **Rust is not a feature, it's a workaround.** Python doesn't expose SIMD intrinsics yet. That's the only reason a compiled language is involved. Rust doesn't manage memory, doesn't hold state, doesn't make decisions. It receives a buffer, runs a vectorized instruction, and returns the result. Any language that can talk to the CPU's vector units would do.
- **All logic stays in Python.** Allocation, lifetimes, ownership, views, shape manipulation — none of this leaks into the compiled side. Python is the brain, Rust is the hand.

The point of these rules is simple: if you took the Rust crate out and replaced it with any other SIMD-capable language, nothing about the engine's design would change. The architecture is Python's, not Rust's.

## The arena

Everything starts with one big block of memory. Instead of allocating space every time a tensor is created, the allocator reserves a single contiguous region upfront using anonymous memory mapping and hands out slices of it on demand. Offsets are page-aligned so the OS can manage the backing pages efficiently.

When a region is no longer needed, it doesn't get returned to the OS. It goes back to a free list — a sorted collection of available regions — so the next allocation can reuse it without any system calls. This is the same pattern behind arena allocators in game engines and database buffer pools. Python can do it too.

The allocator itself is held by a weak reference. It stays alive as long as at least one tensor needs it. The moment the last tensor dies, the allocator gets garbage collected and the memory is automatically released. No manual teardown, no lingering globals — it exists exactly as long as it's useful.

## Frames

A frame is a window into the arena. It knows where its data starts, how large the region is, and what type the values are. When a tensor is created, it gets a frame. When you read or write tensor data, you go through the frame.

Frames can also be narrowed. If you need a sub-region of an existing frame — say, a row of a matrix — you can create a pointer into it. The new frame shares the same underlying region but operates on a smaller window. No data moves.

## Shared ownership

Multiple frames can point to the same region of the arena. This happens naturally when you create views or sub-frame pointers. The question is: who's allowed to write?

The answer is tracked through reference counting on the region's identity. If only one frame references a region, it owns it exclusively and can write freely. The moment a second frame shares that region, both are locked to read-only. This prevents one view from silently corrupting another's data.

There's no borrow checker enforcing this at compile time. It's a runtime check — but it's deterministic, cheap, and entirely built on stdlib primitives.

## Lifecycles

When a tensor dies and its frame is garbage collected, the region needs to come back. This is handled through weak reference finalizers — callbacks that fire when the region's identity object loses all its references.

The finalizer does two things: it hints to the kernel that the pages are no longer needed and can be reclaimed, and it returns the region to the allocator's free list for reuse. No manual cleanup, no explicit free calls. The lifecycle is tied to Python's own reference counting.

This means the allocator is self-healing. Regions flow out on allocation and flow back on garbage collection. The arena stays compact as long as tensors don't outlive their usefulness.

## Views

A tensor is a shape, a set of strides, and a frame. Transpose doesn't move data — it swaps two stride values. Reshape doesn't move data — it recomputes strides. Slicing into a row doesn't move data — it adjusts the offset and narrows the shape.

These operations are essentially free. They produce new tensors that look at the same memory from a different angle. Data only needs to be copied when it has to be laid out contiguously — for example, right before handing a buffer off to a SIMD operation.

## Computation

The engine's arithmetic — addition, multiplication, negation, dot products — is powered by SIMD: single instructions that process multiple floats in parallel.

Python doesn't expose SIMD intrinsics. That's the one gap a compiled language needs to fill — not for its performance, not for its type system, not for its memory safety, but simply because it can talk to the CPU's vector units and Python can't. This project uses Rust, but it could just as easily be Zig, C, or Fortran. The choice doesn't matter. None of Rust's horsepower is being used here. It's a dispatcher — it receives flat buffers, calls the right SIMD instruction, and returns the result. Nothing more.

The crossing between Python and Rust happens through the buffer protocol — CPython's built-in mechanism for sharing memory across language boundaries without copying. Python hands Rust a pointer to the raw bytes living in the arena. Rust hands back a result that Python can read the same way. No serialization, no marshalling, no intermediate copies. The zero-copy philosophy that drives the rest of the engine extends all the way through the FFI boundary.

All the interesting logic — allocation, lifetimes, views, ownership — stays in Python.
