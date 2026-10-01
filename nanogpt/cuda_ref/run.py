#!/usr/bin/env python3
"""Build the hand-written CUDA reference, check it against PyTorch and time it.

  python3 nanogpt/cuda_ref/run.py            # verify + sweep
Needs nvcc (CUDA 12) and an sm_80 GPU (change -arch for another one).
"""
import os, sys, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench_train as b

WORK = os.environ.get("BEND_WORK", "/tmp/bend_train")
os.makedirs(WORK, exist_ok=True)
exe = os.path.join(WORK, "cuda_train")
subprocess.run(["/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_80", "-o", exe, os.path.join(HERE, "cuda_train.cu")], check=True)

m = b.make_model(16)
flat = b.quantize(b.flat_params(m, 16)).astype(np.float32)
pf = os.path.join(WORK, "params.bin")
flat.tofile(pf)

ref = b.run_torch(16, 4, 30, "cpu", print_every=1, flat_override=flat)
out = subprocess.run([exe, pf, "4", "30", "verify"], capture_output=True, text=True).stdout
ls = {int(l.split()[1]): float(l.split()[3]) for l in out.strip().splitlines()}
print("max |loss diff| vs PyTorch over 30 steps:", max(abs(ls[s] - ref["losses"][s]) for s in ls))

for B in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144):
    best = min(float(subprocess.run([exe, pf, str(B), str(20 if B >= 65536 else 50)], capture_output=True, text=True)
                     .stdout.split()[1]) for _ in range(3))
    print(f"B={B:7d}  hand CUDA {best:9.4f} ms/step")
