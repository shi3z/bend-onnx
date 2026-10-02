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

**Milestone 4a (done): generic arrays, arrays of records.** `prelude/arr.bend` has `Arr<&2, A>` for any `Data`
element type (a perfect tree again, so it is shareable and runs on the stock runtime), `Arr.get(-A, d, b, i, z)` and
`Arr.map1..4(~A0.., ~C, ~f, d, b0.., off)`. On the device an `Arr<&2, A>` is a pointer to A (an array of structs when
A is a record), `get` is one load and each `map` is one kernel launch whose result stays on the device, so a pipeline
can hand arrays of records from stage to stage (`tests/arr_rec.bend`: scalar array -> P records -> gathered Q records ->
scalars; matches the stock runtime). All nine tests pass.

**Milestone 4b (done): the nanoGPT training step.** `nanogpt/gen.py` writes the whole step (forward, backward,
weight gradients, Adam) as ordinary Bend: a chain of `Arr.map` stages over per-token records ("tapes"), 17 kernel
launches per step, no heap, no CUDA written by hand. The loss over 30 steps matches PyTorch to 1.4e-5 and every
parameter gradient to <1e-7 (`python3 nanogpt/run.py verify`; `nanogpt/dbg.py` compares gradients).

| B (samples/step) | hand CUDA | PyTorch CUDA | **Bend on bendgpu** | Bend on its own GPU runtime (v4) |
|---:|---:|---:|---:|---:|
| 4096  | 0.95 ms | 3.19 ms | **4.97 ms** | 161 ms |
| 65536 | 14.8 ms | 33.9 ms | **71.3 ms** | 453 ms |

(ms per training step, min of 3, shared A100; the same model, init, data and Adam everywhere.) That is 5x the
hand-written kernel and 1.6-2.1x PyTorch, against 13-30x before. Per stage at B=4096 (ms): the 12 forward/backward
stages take 0.1-0.3 each (2.3 together), the three weight-gradient stages 1.0 each, the chunk reduction 0.3.

What made it fast, in order of effect: (1) a weight-gradient thread owns 16 consecutive parameters and a chunk of
tokens, and a whole warp runs the same job, so the field indices are compile-time constants and only the needed
fields are loaded (64 -> 13.5 ms); (2) more chunks, so there are enough threads (-> 6.8 ms); (3) two-level
reduction of the chunks (-> 5.0 ms); (4) a heap-free program may launch one thread per element instead of the
16384 threads the arenas allow.

```
python3 bendgpu/nanogpt/run.py verify --B 4 --steps 30      # loss curve vs PyTorch
python3 bendgpu/nanogpt/run.py bench --Bs 4096,65536
BG_PROF=1 ./train_binary 20 12289 params.bin                # per-stage times
```

Rules of the checker that shaped the generator (they apply to any program for this back end): a `match` may only
scrutinise parameters and fields, in parameter order, and not after a `let`, so loaded records are passed to a body
def as its first parameters; a `let`-bound value used twice must be `+`.

```
node --experimental-transform-types emit_cuda.mjs prog.bend kern > prog.cu
nvcc -O3 -arch=sm_80 --fmad=false -o prog prog.cu && ./prog 4096
```

Anything outside the subset is rejected with the name of the construct (`unsupported in <def>: ...`), never silently
approximated.

Tests (`tests/run_tests.py`): each program is compiled to CUDA and its outputs are compared, at 10 indices, with the
stock `bend` runtime running the same source (buffers rebuilt there with `Buf.build`). All ten pass: a counted
loop, nested records, a 200-step escape-time loop with `Bool.pick`, a dot product and a matrix-vector product through
buffers, a two-stage pipeline, a list of records, a tree, a pipeline of record arrays and nested records.

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
4. ~~nanoGPT training step within a small factor of `nanogpt/cuda_ref/cuda_train.cu`~~ (done: 5x; the gap is the
   weight-gradient stages, which need shared-memory tile cooperation or multi-row register tiles).
