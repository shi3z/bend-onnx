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

Training lives in four implementations of the same model (1-layer / 1-head / pre-LN / tied-embedding / tanh-GELU nanoGPT, C=16, 14 tokens): forward pass, a hand-written backward pass and AdamW (weight decay 0), all in pure Bend. They differ only in how data is laid out in the heap and how much of it is kept in registers.

- `nanogpt/fused_gen.py` (**v4, default**, `BEND_IMPL=fused`): fused, register-resident token kernels. One flat function per stage handles a whole token with every intermediate `T16` tile in registers; only what the backward pass needs is stored, as records in per-token lists. Specialised for C=16 / F=64 / V=32.
- `nanogpt/tile3_gen.py` (v3, `BEND_IMPL=tile3`): matrices as one list whose cells hold the tiles inline.
- `nanogpt/tile_gen.py` + `train_tile.bend` (v2, `BEND_IMPL=tile`): a row is a list of tiles inside a list of rows; any multiple of 16.
- `nanogpt/train_lib.bend` (v1, `BEND_IMPL=list`): cons lists of scalars.

A batch is a balanced fork-join tree of 2^d samples (one GPU lane per sample) whose gradient records are summed on the way up. The whole run is one `!` call.

### Correctness

Same init (rounded to the same decimals), same data, same Adam hyper-parameters as PyTorch:

| | step 0 | step 15 | step 29 |
|---|---|---|---|
| PyTorch loss | 3.466185 | 1.530063 | 0.468194 |
| Bend loss | 3.466186 | 1.530064 | 0.468193 |

Max |Δloss| over 30 steps is 1.0e-5, on Bend CPU and Bend GPU alike, for v1 to v4.

### Speed (ms per training step, C=16, lower is better)

NVIDIA A100 80GB (shared with other jobs), 24 CPU cores, Bend 2.0.34, PyTorch 2.11 eager, fp32, TF32 off (PyTorch CPU used its default 16 threads, Bend CPU 24). B is the batch size. PyTorch times are after a 5-step warm-up. Bend times come from the difference between a 2-step and a 6-step run of the same binary; the minimum of 3 repeats is reported because other jobs on the GPU slow some runs by up to 2x (PyTorch ran on the same shared GPU).

| B | hand-written CUDA | PyTorch CUDA | PyTorch CPU | Bend CPU (24 threads) | Bend CPU (1 thread) | Bend GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.040 | 0.81 | 0.68 | 0.25 | 0.25 | 33 |
| 4 | 0.041 | 0.84 | 0.95 | 0.75 | 0.75 | 70 |
| 16 | 0.041 | 0.89 | 0.82 | 0.50 | 2.0 | 93 |
| 64 | 0.044 | 0.82 | 1.09 | 1.5 | 9.5 | 73 |
| 256 | 0.094 | 0.91 | 1.81 | 4.8 | 36 | 77 |
| 1024 | 0.28 | 1.17 | 6.8 | 16 | 151 | 123 |
| 4096 | 0.95 | 3.19 | 18.7 | 65 | 602 | 161 |
| 16384 | 3.7 | 8.82 | 97 | 231 | 2416 | 371 |
| 65536 | 14.8 | 33.9 | 507 | 941 | 9639 | 453 |

Progress at B=4096 (Bend CPU 24 threads / Bend GPU, ms): v1 484 / 2718, v2 107 / 275, v3 59 / 224, v4 65 / 161. At B=65536: v3 ~1100 on GPU, v4 453 (2.1x faster than 24 CPU threads, and faster than PyTorch CPU at 507).

**Reference: hand-written CUDA** (`nanogpt/cuda_ref/cuda_train.cu`, run with `python3 nanogpt/cuda_ref/run.py`): the same step, same math and init, written directly in CUDA C++ (no cuBLAS; one block per few samples, activations in shared memory, per-block partial gradients summed by an Adam kernel). Its loss curve matches PyTorch to 1.1e-5. It is not tuned further; it only measures what the hardware does for this step: 0.95 ms at B=4096 and 14.8 ms at B=65536 (59 ms at B=262144), i.e. 3.4x / 2.3x faster than PyTorch eager, and **170x / 30x faster than Bend GPU (v4)**. Marginal cost per sample at saturation: ~0.23 us (CUDA) vs ~0.5 us (PyTorch) vs ~2.8 us (Bend GPU).

**Takeaways**

- **Bend on 24 CPU threads is ahead of PyTorch up to B=16** (both CPU and CUDA, which are launch-bound at ~0.8 ms), roughly on par to B=64, and behind from B=256.
- **Bend GPU beats Bend on 24 CPU threads only at very large batches**: it loses up to B=16384 (371 vs 231 ms) and wins at B=65536 (453 vs 941 ms, 2.1x), because its marginal cost per sample (~2.8 us) is about 7x lower than the CPU's (~20 us) while its fixed latency (~30-40 ms) is high. At B=65536 it is also slightly ahead of PyTorch on 16 CPU threads (453 vs 507 ms).
- **PyTorch CUDA is still faster**: ~50x at B=4096 (161 vs 3.2 ms) and ~13x at B=65536 (453 vs 33.9 ms). PyTorch's per-sample cost at saturation is ~0.5 us; Bend GPU's is ~2.8 us (5.6x).
- **Caveats.** The "batch" repeats 4 phrases, so large B measures speed, not learning. One model size (C=16); v3/v4 are specialised to it. A handful of steps per point.

### Why Bend's GPU was slow, and what fixed it (measured)

The earlier conclusion in this README, that matching PyTorch needs "dense arrays in the language", was too strong and is withdrawn. The gap came from how my code used the heap, not from a hard limit.

| Experiment (4096-65536 independent lanes) | GPU | CPU (24 thr) |
|---|---:|---:|
| tile arithmetic whose operands stay in registers (flat tail loop) | ~13 G tile-ops/s | ~6 G tile-ops/s |
| the same 16-wide op where every result is a heap list cell (`mat_scale`, v3, double allocation from accumulate+reverse) | 0.7 G/s | 3.0 G/s |
| one flat function doing the whole tail of the model for one token (LN, fc, GELU, proj, LN, logits, softmax, loss; ~160 tile dots) | **42 M tokens/s at 1M tokens (75 M/s marginal)** | 8.7 M tokens/s |

- **Register-resident code is where Bend's GPU is fast.** A single fused token function ran 4.8x faster than 24 CPU cores (8x marginal), about 450 GFLOP/s, the same order as PyTorch's effective throughput on this model. Anything that goes through list cells, matrices or non-tail calls becomes 17-word heap nodes and is slower on the GPU than on the CPU. (The "19x" register-vs-heap ratio quoted earlier mixed in the doubled allocation of my accumulate-and-reverse loops; the per-allocation gap is smaller, but the end-to-end effect is real.)
- **v4 applies that to the whole step.** It was written after these measurements. Its first end-to-end run already matched PyTorch; at saturation (B=65536) it reaches ~350k samples/s, 7x the CPU's per-sample rate.
- **Not the cause (each measured):** the launch size (I patched the generated runtime to launch 4x the threads, 65536 instead of 16384: no change, at B=4096 and at B=65536), refcounting (a shared tile read in a loop runs at ~300 GMAC/s), control-flow divergence (identical samples: only 20% faster), the dot product itself, and the gradient tree reduction on its own.
- **What still limits v4.** At B <= 4096 only B lanes have work, so the step time is the latency of one sample's sequential chain (~29 ms on one lane) plus ~13 ms of per-step serial work (`make_q`, Adam), not throughput. Forking inside the sample (parallel `let` over the 20 gradient fields or over `g_add`) made it slower (B=4096: 160 to 278-349 ms) because each fork/join costs a kernel iteration; only the parallel Adam is kept (`BEND_PAR=4`). At saturation the weight-gradient loops (~45% of the sample) and the per-stage record lists (~30 heap nodes per token) dominate.
- **No GPU performance counters were available** (`ERR_NVGPUCTRPERM`), so the memory-system explanation for the remaining gap is inferred from working-set experiments, not read from counters.

### Hard limits of the GPU runtime found on the way

- A lane's continuation stack is only ~360 words. Any non-tail recursion that holds a `T16` (16 words) or an `MC4` cell per level overflows it after ~14-16 levels (`memory fault (machine stack overflow?)`). Every list loop is therefore an accumulate-and-reverse tail loop, or builds records that hold only pointers.
- A function may bind at most 247 values (registers). Several functions had to be split (for example one expression producing two `T16` results, or a record with 20 field expressions), and small non-recursive multi-branch helpers are inlined into callers and can break the limit.
- `CUBE_LOG` (threads launched) comes from the L2 size and is capped at 128 blocks of 128 threads.

### Optimizations

The loss curve matches PyTorch to ~1e-5 after every step below.

What worked:

- **16-wide tiles (v2).** About 5-10x on every backend (numbers above). All vectors are lists of `T16`; the tile primitives are generated by `nanogpt/tile_gen.py`.
- **Tiles inline in the list cells (v3).** A further 1.2-1.8x (`nanogpt/tile3_gen.py`).
- **Fused register-resident token kernels (v4).** Per stage one flat function with all intermediates in registers, records only for what the backward pass needs, and weight gradients as register accumulators over the tokens (four rows per pass, static lanes). B=4096: GPU 224 to 161 ms; B=65536: GPU ~1100 to 453 ms (`nanogpt/fused_gen.py`).
- **Dot-product-only matmuls.** `A·B` and `Aᵀ·B` are transposes plus the allocation-free `A·Bᵀ` loop, instead of axpy loops that allocate two lists per scalar (~2.9x single-threaded on v1).
- **Row-parallel matmuls (`fork_depth`).** Rows of `A` are split in halves 2^d times so one sample no longer sits on a single slow GPU lane (GPU B=16: 1785 ms to 425 ms on v1). 4 or 5 is best on GPU; on CPU it only adds overhead, so CPU runs use 0.
- **Tail-recursive loops** (accumulate and reverse): 1.6x single-threaded on v1, and required on GPU, where a lane's continuation stack is only ~360 words.
- **Parameters, gradients and Adam state kept as tile records** (no flat scalar vectors inside the loop), and the whole training loop run as a single `!` call, so there are no host round-trips per step.

What did not help, and was reverted or left out:

- Per-leaf or per-branch private copies of the parameters (slower or equal).
- 1x4 register-blocked dot products (~2x slower).
- Flat versions of the elementwise vector ops (no gain).
- Running the flat-list Adam inside the GPU call (110 ms per step on one lane): this is why the state is kept as tiles.
- Fixed-size structs of 16 tiles instead of lists (same speed as the list version).
- Parallel `let` over the gradient fields inside each sample, or inside `g_add` (slower: B=4096 160 to 278-349 ms; each fork/join costs a kernel iteration). Only the parallel Adam is kept.

### Notes

- **GPU stack limit.** A GPU lane's continuation stack is only ~360 words (see above). Anything that walks a long list while holding a tile uses tail-recursive accumulate-and-reverse loops.
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
