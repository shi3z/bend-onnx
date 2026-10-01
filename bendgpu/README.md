# bendgpu: a CUDA back end for (a subset of) Bend

Why: in this repo, the same nanoGPT training step runs 170x (B=4096) to 30x (B=65536) slower on Bend's GPU
runtime than a plain hand-written CUDA kernel (see the top-level README). The cause is the execution model
(heap nodes, a 360-word continuation stack, 247 registers per function, fork/join = one kernel iteration),
not the language: code that stays in registers runs at hardware speed even there.

`bendgpu` keeps the **language** (the parser, type/quantity checker and termination checker of
[bendlang/bend](https://github.com/bendlang/bend), used as an untouched library) and replaces the **back end**
for programs that fit a first-order numeric subset: they become plain CUDA (registers, loops, structs),
launched as a parallel map over an index range.

## Status: first slice

```
node --experimental-transform-types emit_cuda.mjs prog.bend kern > prog.cu
nvcc -O3 -arch=sm_80 --fmad=false -o prog prog.cu && ./prog 4096
```

Supported: `U32 F32 Bool Nat` (Nat as a counter); single-constructor records of scalars/records (C structs);
`match` on Bool, Nat and records; tail self-calls (become `for(;;)` loops); calls between subset defs; the Base
primitives (arithmetic, comparisons, `Bool.pick`, `F32.exp/log/tanh/...`) mapped by name; the entry
`def kern(i: U32) -> scalar|record` becomes a kernel over `i in [0, N)`. Anything else is rejected with the name
of the construct (`unsupported in <def>: ...`), never silently approximated.

Tests (`tests/run_tests.py`): each program is compiled to CUDA and its outputs are compared, at 10 indices,
with the output of the stock `bend` runtime on the same source. All pass (a 200-step escape-time loop with `Bool.pick`
and Nat fuel, nested records, a counted loop).

First measurement, same source: the 16-wide `T16` tile loop (`t_dot` chain) that runs on Bend's own GPU runtime at
~6 ms for 4096 x 20000 iterations takes 1.9 ms here (`T16` becomes a 16-float struct in registers, `Nat` a counter).

## Setup

```
git clone https://github.com/bendlang/bend bendgpu/vendor/bend   # or set BEND_SRC=/path/to/bend
node --version            # >= 22 (uses --experimental-transform-types)
```
`bendgpu/vendor/` is git-ignored; only the Bend library is imported, none of its code is copied here.

## Roadmap (each step is a measurable milestone)

1. **Read-only buffers**: a shareable (`Data`) flat array type, `get` compiled to a global/shared load. Needed for weights.
2. **Parallel loops**: `!` on a def over an index range becomes a grid-level loop instead of a fork tree; threads
   cooperate through shared-memory tiles.
3. **Heap-free data structures**: lists/trees of the subset in a per-thread arena, so records-of-lists programs
   (the fused nanoGPT kernels in `nanogpt/fused_gen.py`) compile as they are.
4. **Target**: the nanoGPT training step from `nanogpt/`, within a small factor of `nanogpt/cuda_ref/cuda_train.cu`.
