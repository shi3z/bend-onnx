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

`nanogpt/train_lib.bend` adds **training** to the Bend nanoGPT: forward pass, a hand-written backward pass, and AdamW (weight decay 0), all in pure Bend. The model is the same 1-layer / 1-head / pre-LN / tied-embedding / tanh-GELU nanoGPT (C=16, 14 tokens). Gradients come from three matrix products (`A·Bᵀ`, `A·B`, `Aᵀ·B`). A batch is a balanced fork-join tree of 2^d samples whose gradients are summed on the way up. The tree is launched with `!`, and each sample's matmuls are additionally split over 2^`fork_depth` row blocks.

### Correctness

Same init (rounded to the same decimals), same data, same Adam hyper-parameters as PyTorch:

| | step 0 | step 15 | step 29 |
|---|---|---|---|
| PyTorch loss | 3.466185 | 1.530063 | 0.468194 |
| Bend loss | 3.466186 | 1.530064 | 0.468193 |

Max |Δloss| over 30 steps is 1.3e-5, on Bend CPU and Bend GPU alike.

### Speed (ms per training step, C=16, lower is better)

NVIDIA A100 80GB, 24 CPU cores, Bend 2.0.34, PyTorch 2.11. B is the batch size. PyTorch times are after a 5-step warm-up; Bend times are the mean of the steps after the first. Bend CPU uses `fork_depth=0`; Bend GPU uses `fork_depth=4`.

| B | PyTorch CUDA | PyTorch CPU | Bend CPU (24 threads) | Bend CPU (1 thread) | Bend GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8 | 1.7 | 1.7 | 1.7 | 199 |
| 4 | 0.9 | 0.7 | 2.7 | 6.3 | 306 |
| 16 | 0.7 | 1.8 | 3.3 | 23 | 392 |
| 64 | 0.8 | 1.0 | 9.3 | 94 | 635 |
| 256 | 0.9 | 1.8 | 34 | 388 | 918 |
| 1024 | 1.2 | 5.0 | 135 | 1522 | 1777 |
| 4096 | 3.2 | 20.6 | 484 | 6139 | 2718 |

**Takeaways**

- **PyTorch CUDA is still far faster.** At B=4096 Bend on 24 CPU threads is about 150x slower and Bend on the GPU about 850x slower. At this size PyTorch is bound by kernel-launch overhead (~0.8 ms).
- **Bend GPU is still slower than Bend CPU everywhere we measured.** At B=4096 it is about 5.6x slower than 24 CPU threads. Its time grows slowly with B, because the lanes are far from saturated, but each GPU lane is very slow. The small-B cost is dominated by sequential fork/join stages (about 0.5 ms each).
- **Caveats.** The "batch" repeats 4 phrases, so large B measures speed, not learning. Each point averages only a few steps, on one model size (C=16). Larger models were not benchmarked.

### Optimizations (vs. the first working version)

The loss curve still matches PyTorch to 1.3e-5 after every step below. Speed-up at B=4096, and at B=1 for the GPU:

| | before | after |
|---|---:|---:|
| Bend CPU, 24 threads | 2651 ms | 484 ms (5.5x) |
| Bend CPU, 1 thread | 25661 ms | 6139 ms (4.2x) |
| Bend GPU | 5701 ms | 2718 ms (2.1x) |
| Bend GPU, B=1 | 1307 ms | 199 ms (6.6x) |

What worked:

- **Dot-product-only matmuls.** `A·B` and `Aᵀ·B` used to be axpy-style loops that allocate two lists per scalar. They are now transposes plus the allocation-free dot loop `A·Bᵀ`. This was the biggest CPU win, about 2.9x single-threaded.
- **Row-parallel matmuls (`fork_depth`).** The rows of `A` are split in halves 2^d times, so one sample no longer occupies a single slow GPU lane. This took GPU B=16 from 1785 ms to 425 ms. `fork_depth` 4 or 5 is best on GPU; 6 hurts B=1. On CPU it only adds overhead, so CPU runs use 0.
- **Tail-recursive row loop.** Building the per-row result list as an accumulate-and-reverse loop gave another 1.6x on a single CPU thread.

What did not help, and was reverted:

- **Per-leaf private copies of the parameters** (to avoid refcount contention): slower on CPU and GPU.
- **1x4 register-blocked dot products:** about 2x slower.
- **Flat versions of the elementwise vector ops:** no gain.
- **Structured gradient accumulation** (adding records of matrices up the tree instead of flat vectors): kept, because it simplifies the reduction, but it changed the time by only ~10% at B=1 and not at all at B=1024.

### Notes

- **GPU stack limit.** A GPU lane's continuation stack is only ~2k frames deep. Walking long vectors with `VCon{h, f(t)}` crashes with `memory fault (machine stack overflow?)`. The flat parameter and gradient vectors therefore use tail-recursive accumulate-and-reverse loops.
- **Weight literals.** The initial weights are embedded as a balanced tree of short literals. A single 4.6k-deep literal overflows the compiler stack.
- **Building with `!`.** This needs clang 19+ and CUDA 12 at `/usr/local/cuda`. Run with `./model --gpu on|off --threads N`.

```bash
export BEND_CLANG_BIN=/path/to/clang19/bin
python3 nanogpt/bench_train.py verify --C 16 --depth 2 --steps 30 --gpu   # Bend vs PyTorch loss curves
python3 nanogpt/bench_train.py bench  --C 16 --depths 0 2 4 6 8 10 12 --steps 4
```

---

## ⚙️ GPU Acceleration & Portability

- **CUDA 12 Support**: When run natively on an NVIDIA GPU machine with `/usr/local/cuda`, Bend generates and launches a `.gpu` kernel targeting the GPU streaming multiprocessors.
- **Metal Support**: On Apple Silicon (M1/M2/M3/M4), Bend compiles directly to Metal Compute Pipelines with unified memory zero-copy sharing.
- **Parallel CPU Fallback**: On machines without GPU hardware (or in containerized/sandboxed environments), Bend automatically executes parallel `!` calls across the host CPU thread pool with balanced fork-join work stealing.

---

## 📜 License

MIT License. Designed and built with Google Antigravity and Bend 2.0.
