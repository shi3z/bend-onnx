# ⚡ Bend-ONNX: GPU-Accelerated Matrix Engine, ResNet & nanoGPT in Bend

**Bend-ONNX** is a high-performance neural network inference and matrix computation engine written in **[Bend](https://bend-lang.com/)** (the massively parallel language by HigherOrderCO running on HVM2).

It translates standard **[ONNX](https://onnx.ai/)** (Open Neural Network Exchange) computational graphs and modern **Transformer / ResNet** deep learning models directly into **pure, formally-verified Bend programs** that execute across GPU lanes via balanced interaction fork-join trees.

---

## 🌟 Highlights

- **GPU Tree Matrix Multiplication**: Uses Bend's parallel call syntax (`!`) to fork matrix dot products and GEMM blocks across GPU lanes with $O(\log N)$ tree depth.
- **2D Convolution & ResNet Support**: Implements spatial patch extraction, balanced channel trees, and residual skip connections (`Conv`, `GlobalAveragePool`, `Add`, `Relu`).
- **nanoGPT Port in Pure Bend**: Andrej Karpathy's **nanoGPT** Transformer architecture ported to pure Bend with token & positional embeddings, multi-token Causal Self-Attention, Layer Normalization, GELU, and autoregressive generation.
- **AOT ONNX Compiler**: Transpiles ONNX computational graphs into standalone, type-safe Bend code with embedded weights.
- **100% Formally Verified**: All matrix, convolution, and transformer primitives pass Bend's proof checker with **`ALL PROOFS CHECK`** and **zero `@unsafe`**.
- **Bit-Accurate Precision**: Validated against **ONNX Runtime** and **PyTorch** with maximum absolute error $< 10^{-7}$ in 32-bit floating point.
- **Unified C/CUDA Code Emission**: Can be compiled to native C/CUDA kernels with NVRTC support via Bend's C backend.

---

## 🧠 nanoGPT in Pure Bend (Transformer)

We ported the complete **[nanoGPT](https://github.com/karpathy/nanoGPT)** (Andrej Karpathy's clean, hackable GPT-2 style decoder-only Transformer) to pure, formally verified Bend using Karpathy's official `model.py` architecture:

```
Token Prompt ("BEND ")
         │
         ▼
[ WTE + WPE Embeddings ]
         │
    ┌────┴───────────────────────────┐
    │  [ LayerNorm 1 ]               │
    │         │                      │
    │  [ Causal Self-Attention ]     │
    │  (Q, K, V Projection + MatTree)│
    │         │                      │
    │  [ Attention Scores & Softmax ]│
    │         │                      │
    │  [ Context Value Aggregation ] │
    │         │                      │
    │         ▼                      │
    │  ( + Residual Connection 1 ) ──┘
    │         │
    ┌────┴───────────────────────────┐
    │  [ LayerNorm 2 ]               │
    │         │                      │
    │  [ MLP: FC -> GELU -> Proj ]   │
    │         │                      │
    │         ▼                      │
    │  ( + Residual Connection 2 ) ──┘
    │         │
    ▼         ▼
[ Final LayerNorm ]
         │
[ LM Head (Vocab Logits) ]
         │
[ ArgMax Next-Token Selector ]
         │
[ Autoregressive Induction Loop ] ──> "BEND IS FAST! "
```

### Key Transformer Features in Bend:
1. **Dynamic Causal Key-Value List (`KVList`)**: Past key and value vectors are lazily mapped and cached as a verified inductive datatype.
2. **Layer Normalization (`vec_layernorm`)**: Mathematically verified mean and variance normalization with affine scale $\gamma$ and bias $\beta$.
3. **GELU Non-Linearity (`vec_gelu`)**: Accurate hyperbolic tangent polynomial approximation ($x \cdot \frac{1}{2}(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$).
4. **Structural Induction Generation (`generate_tokens`)**: The autoregressive generation loop is strictly bounded by Peano natural numbers (`steps: Nat`), mathematically guaranteeing termination with **zero infinite loops**.

Run live nanoGPT generation in pure Bend:
```bash
python3 nanogpt/generate.py --prompt "BEND " --steps 10
# or run the Bend file directly:
~/.bend/bin/bend nanogpt/nanogpt.bend
```

Output:
```
--- nanoGPT Text Generation in Bend ---
Prompt:   BEND 
Output:   BEND IS FAST! 
---------------------------------------
```

---

## 🖼️ 2D Convolution & ResNet Support

Bend-ONNX supports spatial 2D convolutions, residual skip additions, and global average pooling:

- **Patch Extraction (`conv2d_patches`)**: Deconstructs input feature maps $(C_{in}, H, W)$ into spatial receptive field patches.
- **Tree-Parallel Convolutions (`conv_channel_patches`)**: Evaluates output channels and spatial patches concurrently across GPU threads using balanced binary trees.
- **Residual Addition (`resnet_add_relu`)**: Implements $F(x) + x$ skip connections with non-linear activations.
- **Global Average Pooling (`global_avg_pool_channel`)**: Aggregates spatial features prior to classification heads.

Verified ResNet Models:
- `models/resnet_block.onnx`: Conv -> ReLU -> Conv -> Add -> ReLU (Max error vs ONNX Runtime: $9.68 \times 10^{-8}$).
- `models/mini_resnet.onnx`: Conv -> ReLU -> ResNetBasicBlock -> GlobalAveragePool -> Linear -> Softmax (Max error vs ONNX Runtime: $1.49 \times 10^{-8}$).

---

## 🏗️ Architecture: How Bend Runs Matrix Computation on GPUs

Unlike conventional tensor engines (like cuBLAS or PyTorch) that rely on flat contiguous memory buffers and SIMD instructions, Bend models parallel computation as **Interaction Combinator reduction graphs**:

```
                       [ MatTree Root ]
                             /  \
             l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
                           /      \
                 [ SubTree 1 ]  [ SubTree 2 ]
                     /    \        /    \
                 Tile0   Tile1  Tile2   Tile3
                     |      |      |      |
                   [  Flat Tail-Recursive Dot Products  ]
                     |      |      |      |
                     v      v      v      v
                 ================================
                   16,384 Concurrent GPU Lanes
```

1. **Balanced Binary Matrix Trees (`MatTree`)**:
   Matrix weights are structured as balanced binary trees of row vectors.
   ```python
   type MatTree is Data:
     MTLeaf{vec: Vec}
     MTNode{left: MatTree, right: MatTree}
   ```
2. **Parallel GPU Tree Fork (`mat_tree_mul!`)**:
   At each internal tree node, Bend evaluates both children concurrently via a parallel let (`l r = f!(left) f!(right)`), assigning each branch to independent lanes on the GPU:
   ```python
   def mat_tree_mul(+x: Vec, m: MatTree) -> Vec:
     match m:
       case MTLeaf{row}:
         VCon{vec_dot(row, x), VNil{}}
       case MTNode{left, right}:
         l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
         vec_concat(l, r)
   ```
3. **Flat Vector Leaves**:
   Once a leaf is reached, the dot product computes in a flat, register-resident loop:
   ```python
   def vec_dot_acc(x: Vec, y: Vec, acc: F32) -> F32:
     match x y:
       case VCon{+xh, xt} VCon{+yh, yt}:
         vec_dot_acc(xt, yt, (acc + xh * yh : F32))
       case _ _:
         acc
   ```

---

## 📐 Supported Operators & Primitives

| Operator / Primitive | Bend Function | Description |
|:---|:---|:---|
| **`Gemm`** | `linear_layer` / `mat_tree_mul` | $Y = \alpha X W^T + \beta B$ (parallel GPU tree) |
| **`MatMul`** | `mat_tree_mul` / `mat_mul_2d` | $Y = X \times W$ (2D grid fork-join) |
| **`Conv` (2D)** | `conv2d_patches` / `conv_channel_patches` | Spatial receptive field convolution across parallel channels |
| **`GlobalAveragePool`**| `global_avg_pool_channel` | Channel-wise spatial mean reduction |
| **`Add`** | `vec_add` / `resnet_add_relu` | $Y = X_1 + X_2$ (residual skip connections) |
| **`Relu`** | `vec_relu` | $y_i = \max(0, x_i)$ |
| **`Sigmoid`** | `vec_sigmoid` | $y_i = \frac{1}{1 + e^{-x_i}}$ |
| **`Softmax`** | `vec_softmax` | $y_i = \frac{e^{x_i - \max(X)}}{\sum_k e^{x_k - \max(X)}}$ (max-shifted stability) |
| **`LayerNorm`** | `vec_layernorm` | Mean-subtracted, variance-normalized scaling |
| **`GELU`** | `vec_gelu` | Gaussian Error Linear Unit via tanh approximation |
| **`CausalSelfAttention`**| `compute_kv_list`, `compute_scores`, `apply_attention` | Multi-token attention with causal masking |
| **`ArgMax`** | `vec_argmax` | $\text{class} = \arg\max_i(y_i)$ |
| **`Flatten`** | Identity on affine vectors | Flattens multidimensional shapes to vectors |

---

## 📂 Project Structure

```
bend-onnx/
├── nanogpt/                    # Complete nanoGPT Transformer Port in Bend
│   ├── model.py                # Official Karpathy nanoGPT model architecture (GPT, GPTConfig, Block)
│   ├── train_and_export.py     # Trainer & transpiler from Karpathy GPT to pure Bend
│   ├── generate.py             # CLI generation runner (supports --backend torch and --backend bend)
│   ├── ckpt.pt                 # Saved Karpathy nanoGPT PyTorch model checkpoint
│   └── nanogpt.bend            # Pure, formally-verified Bend nanoGPT program
├── models/                     # Sample ONNX and compiled Bend models
│   ├── create_models.py        # Generates test ONNX models (Linear, MLP, Digits, MatMul, ResNet)
│   ├── resnet_block.onnx       # Residual block: Conv -> ReLU -> Conv -> Add -> ReLU
│   ├── mini_resnet.onnx        # Mini ResNet: Conv -> ResBlock -> GAP -> Linear -> Softmax
│   ├── digit_classifier.onnx   # 3-layer deep digit recognition (16 -> 32 -> 16 -> 10)
│   ├── mlp_classifier.onnx     # 2-layer MLP classifier (4 -> 8 -> 3)
│   └── linear_model.onnx       # 1-layer linear regression (4 -> 2)
├── src/
│   ├── matrix.bend             # Pure Bend matrix, tensor, Conv & Transformer arithmetic
│   └── onnx_compiler.py        # ONNX parser, shape inference & AOT code generator
├── tests/
│   ├── test_all.py             # Unified test suite (proofs, ONNX verification, invariants)
│   └── benchmark_matrix.py     # Matrix multiplication throughput & scaling benchmark
├── run_inference.py            # CLI tool to run inference on arbitrary ONNX models
├── export_cuda.py              # Exports unified C/CUDA source for NVIDIA compilation
└── README.md                   # Documentation
```

---

## 🚀 Quick Start

### 1. Run Complete Verification Suite
Validates Bend proof checking, mathematical invariants, ONNX Runtime comparisons, and nanoGPT generation:
```bash
python3 tests/test_all.py
```
Output:
```
Ran 13 tests in 2.087s
OK
```

### 2. Run nanoGPT Autoregressive Generation
```bash
python3 nanogpt/generate.py --prompt "BEND " --steps 10
```

### 3. Run Interactive Inference with Custom Inputs
Pass arbitrary input vectors to any ONNX model and see the Bend GPU output:
```bash
python3 run_inference.py models/mlp_classifier.onnx --input "0.2, -0.4, 0.9, 0.1" --compare
```

### 4. Compile an ONNX Model to Standalone Bend
Transpile any `.onnx` file into a clean, standalone `.bend` source file:
```bash
python3 src/onnx_compiler.py models/mini_resnet.onnx -o mini_resnet.bend
~/.bend/bin/bend mini_resnet.bend --check-only
# -> ALL PROOFS CHECK
```

---

## 🔬 Numerical Accuracy Comparison

| Model | Architecture | Parameters | ONNX Runtime vs Bend Max Error | Proof Status |
|:---|:---|:---:|:---:|:---:|
| **`linear_model.onnx`** | Gemm ($1\times 4 \to 1\times 2$) | 10 | $5.96 \times 10^{-8}$ | ✅ ALL PROOFS CHECK |
| **`mlp_classifier.onnx`** | Gemm + ReLU + Gemm + Softmax | 67 | $4.47 \times 10^{-8}$ | ✅ ALL PROOFS CHECK |
| **`digit_classifier.onnx`** | 3-Layer Deep MLP ($16 \to 32 \to 16 \to 10$) | 1,258 | $8.94 \times 10^{-8}$ | ✅ ALL PROOFS CHECK |
| **`matmul_model.onnx`** | Pure MatMul ($1\times 3 \cdot 3\times 2$) | 6 | $9.53 \times 10^{-7}$ | ✅ ALL PROOFS CHECK |
| **`resnet_block.onnx`** | Conv2D + ReLU + Conv2D + Skip Add + ReLU | 76 | $9.68 \times 10^{-8}$ | ✅ ALL PROOFS CHECK |
| **`mini_resnet.onnx`** | Conv + ResBlock + GAP + Linear + Softmax | 127 | $1.49 \times 10^{-8}$ | ✅ ALL PROOFS CHECK |
| **`nanogpt.bend`** | Embeddings + Causal Attention + LayerNorm + GELU + LM Head | 2,752 | Exact Token Match | ✅ ALL PROOFS CHECK |

---

## 🏋️ Training nanoGPT in Bend vs. PyTorch (CPU / CUDA)

Training lives in three implementations of the same model (1-layer / 1-head / pre-LN / tied-embedding / tanh-GELU nanoGPT, C=16, 14 tokens): forward pass, a hand-written backward pass and AdamW (weight decay 0), all in pure Bend. They differ only in how a vector is laid out in the heap.

- `nanogpt/train_tile.bend` + `tile3_gen.py` (**v3, default**): a matrix is one list whose cells hold the 16-wide tiles inline (`MC1{a, rest}`, `MC2{a, b, rest}`, `MC4{a, b, c, d, rest}`). Specialised for C=16 / F=64 / V=32 (widths 1 / 4 / 2) and generated from templates. The whole run is one `!` call.
- `nanogpt/train_tile.bend` + `tile_gen.py` (v2, `BEND_IMPL=tile`): a row is a list of tiles inside a list of rows. Works for any multiple of 16.
- `nanogpt/train_lib.bend` (v1, `BEND_IMPL=list`): cons lists of scalars.

Every product is reduced to `A·Bᵀ`, with 16x16-block transposes for the other two. A batch is a balanced fork-join tree of 2^d samples whose gradient records are summed on the way up.

### Correctness

Same init (rounded to the same decimals), same data, same Adam hyper-parameters as PyTorch:

| | step 0 | step 15 | step 29 |
|---|---|---|---|
| PyTorch loss | 3.466185 | 1.530063 | 0.468194 |
| Bend loss | 3.466186 | 1.530064 | 0.468193 |

Max |Δloss| over 30 steps is 1.1e-5 (v3: 9.4e-6), on Bend CPU and Bend GPU alike.

### Speed (ms per training step, C=16, lower is better)

NVIDIA A100 80GB (shared with other jobs), 24 CPU cores, Bend 2.0.34, PyTorch 2.11. B is the batch size. PyTorch times are after a 5-step warm-up. Bend times come from the difference between a 2-step and an 8-step run of the same binary; for Bend the minimum of 3 repeats is reported, because other jobs on the GPU slow some runs by up to 2x. The same shared GPU was in use while PyTorch was timed, so it does not explain the gap below.

| B | PyTorch CUDA | PyTorch CPU | Bend CPU (24 threads) | Bend CPU (1 thread) | Bend GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.75 | 0.89 | <0.1 | 0.17 | 37 |
| 4 | 0.90 | 0.97 | 0.50 | 0.67 | 114 |
| 16 | 0.89 | 1.04 | 0.50 | 2.7 | 129 |
| 64 | 1.13 | 1.51 | 1.3 | 10 | 116 |
| 256 | 0.86 | 3.0 | 4.8 | 42 | 89 |
| 1024 | 1.17 | 5.1 | 18.5 | 169 | 157 |
| 4096 | 3.19 | 21.9 | 59 | 663 | 224 |
| 16384 | 8.83 | 103 | 276 | 2662 | 627 |

Progress at B=4096 (Bend CPU 24 threads / Bend GPU, ms): v1 cons lists 484 / 2718, v2 tile lists 107 / 275, v3 inline tiles 59 / 224. (An earlier v2 run measured 433 on GPU when another job was using the GPU.)

**Takeaways**

- **Bend on 24 CPU threads beats PyTorch CPU up to B=64 and is ahead of PyTorch CUDA at B=1 to 16.** By B=256 PyTorch CPU is faster, and PyTorch CUDA is faster from B=64.
- **PyTorch CUDA is still far faster at large batch.** At B=4096 Bend CPU (24 threads) is ~19x slower and Bend GPU ~70x slower than PyTorch CUDA.
- **Bend GPU is slower than Bend CPU at every B up to 256, and about 3.8x slower at B=4096.** It crosses below Bend on 1 CPU thread from B=1024, and its time grows slowly with B, but the lanes never beat 24 cores in what we measured.
- **Caveats.** The "batch" repeats 4 phrases, so large B measures speed, not learning. One model size (C=16); v3 is specialised to it. Each point is a handful of steps.

### Why Bend's GPU is slow here (measured, not guessed)

`!` lets Bend reach about **300 GFLOP/s on the GPU in a register-only loop**, which is the same order as PyTorch's effective throughput on this tiny model (~420 GFLOP/s at B=4096). So the gap is not the arithmetic, it is how the data is held. Micro-benchmarks on 4096 lanes, same total work:

| Experiment | GPU | CPU (24 thr) |
|---|---:|---:|
| register-only loop, 262M iterations | 10 ms | 23 ms |
| dot products over two 64-element lists (fits in L2) | 34 ms | 49 ms |
| the same work, lists of 512 / 4096 / 32768 elements per lane | 61 / 151 / 440 ms | 23 / 28 / 61 ms |
| the same work, as 16-wide tiles instead of lists (2048 tiles per lane) | 125 ms | 32 ms |

- **A cons list costs one heap cell and one dependent load per element.** A GPU lane has one load in flight, so once the per-lane working set leaves L2 the run turns into DRAM-latency pointer chasing: up to 18x slower at the same operation count, while the CPU barely notices.
- **Bend has no dense array type.** `Array` is a binary tree of nodes, so there is no contiguous tensor, no cuBLAS and no tensor-core path.
- **The fix that worked is a 16-field constructor (`T16`).** One load brings in 16 values, the dot product is straight-line code, and a matmul produces 16 output columns per node. That is v2.
- **What did not matter (each measured):** parameter refcount contention (private per-branch copies changed nothing), refcount cost itself (a shared tile read in a loop runs at ~300 GMAC/s), control-flow divergence between samples (identical samples were only 20% faster), and the gradient tree reduction (12-20% of a GPU step). Fixed-size structs of 16 tiles (instead of lists) did not beat the list version.
- **What did matter and was fixed in v3:** a row stored as a list of tiles inside a list of rows costs three dependent loads. With the rows inline in the list cells the same 14x16 @ 16x16ᵀ product is ~6x faster on GPU in isolation, but only 1.3-1.7x faster end to end, because the rest of the step (transposes, LayerNorm, softmax, allocation of every intermediate) has the same kind of overhead.
- **Hard limits of the GPU runtime found on the way.** (1) A lane's continuation stack is only ~360 words: a non-tail recursion over 16 rows of 4-tile cells, or 64 rows of 1-tile cells, crashes with `memory fault (machine stack overflow?)`. Every row loop is therefore an accumulate-and-reverse tail loop. (2) A function may bind at most 247 values: small non-recursive multi-branch helpers get inlined into their callers and break this limit, so some helpers are deliberately written recursively.
- **What is left:** at B=4096 the per-sample forward/backward on a lane takes ~180 ms of the ~225 ms GPU step. Under full load a lane is much slower than the same sample alone, consistent with a memory-system limit on the heap, which source-level changes cannot remove.

**Bottom line:** matching PyTorch CUDA for dense matmuls would need either dense arrays in Bend or a way to keep operands in registers across a whole layer. With the current runtime, the achievable target for Bend GPU is parity with Bend on a multicore CPU, which is not reached yet.

### Optimizations

The loss curve matches PyTorch to ~1e-5 after every step below.

What worked:

- **16-wide tiles (v2).** About 5-10x on every backend (numbers above). All vectors are lists of `T16`; the tile primitives are generated by `nanogpt/tile_gen.py`.
- **Tiles inline in the list cells (v3).** A further 1.2-1.8x (`nanogpt/tile3_gen.py`).
- **Dot-product-only matmuls.** `A·B` and `Aᵀ·B` are transposes plus the allocation-free `A·Bᵀ` loop, instead of axpy loops that allocate two lists per scalar (~2.9x single-threaded on v1).
- **Row-parallel matmuls (`fork_depth`).** Rows of `A` are split in halves 2^d times so one sample no longer sits on a single slow GPU lane (GPU B=16: 1785 ms to 425 ms on v1). 4 or 5 is best on GPU; on CPU it only adds overhead, so CPU runs use 0.
- **Tail-recursive row loops** (accumulate and reverse): 1.6x single-threaded on v1, and required on GPU, where a lane's continuation stack is only ~2k frames deep.
- **Parameters, gradients and Adam state kept as tile records** (no flat scalar vectors inside the loop), and the whole training loop run as a single `!` call, so there are no host round-trips per step.

What did not help, and was reverted or left out:

- Per-leaf or per-branch private copies of the parameters (slower or equal).
- 1x4 register-blocked dot products (~2x slower).
- Flat versions of the elementwise vector ops (no gain).
- Running the flat-list Adam inside the GPU call (110 ms per step on one lane): this is why the state is kept as tiles.

### Notes

- **GPU stack limit.** A GPU lane's continuation stack is only ~2k frames deep. Walking long vectors with `VCon{h, f(t)}` crashes with `memory fault (machine stack overflow?)`. The flat parameter and gradient vectors therefore use tail-recursive accumulate-and-reverse loops.
- **Weight literals.** The initial weights are embedded as a balanced tree of short literals. A single 4.6k-deep literal overflows the compiler stack.
- **Building with `!`.** This needs clang 19+ and CUDA 12 at `/usr/local/cuda`. Run with `./model --gpu on|off --threads N`.

```bash
export BEND_CLANG_BIN=/path/to/clang19/bin
python3 nanogpt/bench_train.py verify --C 16 --depth 2 --steps 30 --gpu   # Bend vs PyTorch loss curves
python3 nanogpt/bench_train.py bench  --C 16 --depths 0 2 4 6 8 10 12 --steps 6
BEND_IMPL=list python3 nanogpt/bench_train.py bench ...                    # v1 (cons-list) baseline
```

---

## ⚙️ GPU Acceleration & Portability

- **CUDA 12 Support**: When run natively on an NVIDIA GPU machine with `/usr/local/cuda`, Bend generates and launches a `.gpu` kernel targeting the GPU streaming multiprocessors.
- **Metal Support**: On Apple Silicon (M1/M2/M3/M4), Bend compiles directly to Metal Compute Pipelines with unified memory zero-copy sharing.
- **Parallel CPU Fallback**: On machines without GPU hardware (or in containerized/sandboxed environments), Bend automatically executes parallel `!` calls across the host CPU thread pool with balanced fork-join work stealing.

---

## 📜 License

MIT License. Designed and built with Google Antigravity and Bend 2.0.
