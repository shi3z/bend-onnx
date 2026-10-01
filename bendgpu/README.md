# bendgpu: a CUDA back end for (a subset of) Bend

Why: in this repo, the same nanoGPT training step runs 170x (B=4096) to 30x (B=65536) slower on Bend's GPU
runtime than a plain hand-written CUDA kernel (see the top-level README). The cause is the execution model
(heap nodes, a 360-word continuation stack, 247 registers per function, fork/join = one kernel iteration),
not the language: code that stays in registers runs at hardware speed even there.

`bendgpu` keeps the **language** (the parser, type/quantity checker and termination checker of
[bendlang/bend](https://github.com/bendlang/bend), used as an untouched library) and replaces the **back end**
for programs that fit a first-order numeric subset: they become plain CUDA (registers, loops, structs),
launched as a parallel map over an index range.

## Status

**Milestone 0 (done): scalars, records, loops.** `U32 F32 Bool Nat` (Nat as a counter); single-constructor records
of scalars/records (C structs); `match` on Bool, Nat and records; tail self-calls (become `for(;;)` loops); calls
between subset defs; the Base primitives (arithmetic, comparisons, `Bool.pick`, `F32.exp/log/tanh/...`) mapped by name.

**Milestone 1 (done): read-only buffers.** `prelude/buf.bend` defines `Buf`, a perfect binary tree of `F32` that is
`Data` (shareable by every thread; Base's `Array` is affine and cannot be read by many lanes), and
`Buf.get(d, b, i)`. It is ordinary Bend, so it type-checks and runs on the stock runtime, which is the reference
semantics; the back end compiles a `Buf` to a device pointer and `get` to one load. An entry
`def kern(+i: U32, +w: B.Buf, +x: B.Buf) -> value` takes its buffers from the program itself: for each buffer
parameter `w` the file defines `w_depth() -> Nat` (the buffer holds 2^depth values) and
`w_init(j: U32) -> F32` (element j; `j` used once). The generated host code fills the buffers with an init kernel.

**Milestone 2 (done, first half): parallel maps and device pipelines.** `Buf.map1..4(~f, d, bufs.., 0)` (in
`prelude/buf.bend`, plain Bend: a fork tree on the stock runtime) is the parallel loop: element i of the result is
`f(i, bufs..)`, for i in [0, 2^d). The back end turns each call into one kernel launch and keeps the result in device
memory for the next stage. An entry def from buffers to a buffer (`def pipe(+w1: B.Buf, +x: B.Buf, ..) -> B.Buf`) whose
body is a chain of `+h : B.Buf = B.Buf.map2(~s1, 5n, w1, x, 0)` lets becomes a host program that launches them in order
(`tests/pipe_mlp.bend`: `h = tanh(W1 x)` then `y = W2 h`, matches the stock runtime). Not done yet from the original
plan: threads of a block cooperating through shared-memory tiles.

**Milestone 3 (done): heap data structures.** Lists, trees and any multi-constructor or recursive ADT, including
parametric ones (`List<&2, P>`: the type arguments are substituted into the constructor types). A record
(single constructor, non-recursive) is a C struct held by value; every other ADT is a tagged node in a per-thread bump
arena (`BG_ARENA_KB`, default 64 KB, reset for each item) held by pointer, with the single nullary constructor
(`Nil`) as `nullptr`. Patterns work on a queue of pending values, so nested patterns
(`case Con{P{a, b}, t}`) compile as they are lowered by the checker; non-tail recursion is a plain device call
(`BG_STACK_KB`); constructors take their type from the expected type (the return type, a parameter type or a
`{x : T}` annotation). Kernels run as a grid-stride loop over at most 16384 resident threads, each with its own
arena. `tests/list.bend` (records in a list built by non-tail recursion, nested pattern fold) and `tests/tree.bend`
(tree with data at the nodes) match the stock runtime.

```
node --experimental-transform-types emit_cuda.mjs prog.bend kern > prog.cu
nvcc -O3 -arch=sm_80 --fmad=false -o prog prog.cu && ./prog 4096
```

Anything outside the subset is rejected with the name of the construct (`unsupported in <def>: ...`), never silently
approximated.

Tests (`tests/run_tests.py`): each program is compiled to CUDA and its outputs are compared, at 10 indices, with the
stock `bend` runtime running the same source (buffers rebuilt there with `Buf.build`). All eight pass: a counted
loop, nested records, a 200-step escape-time loop with `Bool.pick`, a dot product and a matrix-vector product through
buffers, a two-stage pipeline, a list of records and a tree.

Measurements, same source:
- 16-wide `T16` tile loop (`t_dot` chain, 4096 x 20000 iterations): ~6 ms on Bend's own GPU runtime, 1.9 ms here
  (`T16` is a 16-float struct in registers, `Nat` a counter).
- `bench/matvec.bend` (4096 x 16 weights read through `Buf.get`, 65536 threads, one row each): 229 ms on Bend's GPU
  runtime (257 ms on 24 CPU threads, where `Buf.get` walks a depth-16 tree per element), **0.008 ms** here, same sum.

## Setup

```
git clone https://github.com/bendlang/bend bendgpu/vendor/bend   # or set BEND_SRC=/path/to/bend
node --version            # >= 22 (uses --experimental-transform-types)
```
`bendgpu/vendor/` is git-ignored; only the Bend library is imported, none of its code is copied here.

## Roadmap (each step is a measurable milestone)

1. ~~Read-only buffers~~ (done, see above).
2. ~~Parallel loops~~ (maps and pipelines done; shared-memory tile cooperation still open).
3. ~~Heap data structures~~ (done: per-thread arena).
4. **Target**: the nanoGPT training step from `nanogpt/`, within a small factor of `nanogpt/cuda_ref/cuda_train.cu`.
