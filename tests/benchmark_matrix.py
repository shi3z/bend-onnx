#!/usr/bin/env python3
"""
Benchmark for Bend GPU/Parallel Tree Matrix Multiplication.
Generates balanced NxN matrices and evaluates multiplication execution times.
"""

import sys
for p in [
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/site-packages",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import sys
import time
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

def generate_benchmark_bend(dim: int, temp_file: str = "/tmp/bench_mat.bend"):
    """Generates a Bend program with an NxN matrix multiplied by an N vector."""
    np.random.seed(42)
    A = np.random.uniform(-1.0, 1.0, (dim, dim)).astype(np.float32)
    x = np.random.uniform(-1.0, 1.0, (dim,)).astype(np.float32)
    
    from src.onnx_compiler import build_mat_tree, format_vec
    
    code = f"""import Base

type Vec is Data:
  VNil{{}}
  VCon{{head: F32, tail: Vec}}

def vec_dot_acc(x: Vec, y: Vec, acc: F32) -> F32:
  match x y:
    case VCon{{+xh, xt}} VCon{{+yh, yt}}:
      vec_dot_acc(xt, yt, (acc + xh * yh : F32))
    case _ _:
      acc

def vec_dot(x: Vec, y: Vec) -> F32:
  vec_dot_acc(x, y, 0.0)

def vec_concat(x: Vec, y: Vec) -> Vec:
  match x:
    case VNil{{}}:
      y
    case VCon{{h, t}}:
      VCon{{h, vec_concat(t, y)}}

type MatTree is Data:
  MTLeaf{{vec: Vec}}
  MTNode{{left: MatTree, right: MatTree}}

# GPU Parallel Tree Matrix-Vector Multiplication:
def mat_tree_mul(+x: Vec, m: MatTree) -> Vec:
  match m:
    case MTLeaf{{row}}:
      VCon{{vec_dot(row, x), VNil{{}}}}
    case MTNode{{left, right}}:
      l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
      vec_concat(l, r)

def vec_len(v: Vec) -> U32:
  match v:
    case VNil{{}}:
      0
    case VCon{{_, t}}:
      (1 + vec_len(t) : U32)

def matrix_weights() -> MatTree:
  {build_mat_tree(A.tolist())}

def input_vector() -> Vec:
  {format_vec(x.tolist())}

def main() -> IO(Unit):
  do IO<Unit>:
    x : Vec = input_vector()
    m : MatTree = matrix_weights()
    +out : Vec = mat_tree_mul(x, m)
    IO.print("BENCHMARK_DONE: " ++ U32.show(vec_len(out)))
"""
    with open(temp_file, "w") as f:
        f.write(code)
    return A, x

def benchmark_dimension(dim: int):
    temp_file = f"/tmp/bench_mat_{dim}.bend"
    A, x = generate_benchmark_bend(dim, temp_file)
    
    # Measure Bend execution time
    t0 = time.perf_counter()
    proc = subprocess.run([BEND_BIN, temp_file], capture_output=True, text=True)
    t1 = time.perf_counter()
    
    if proc.returncode != 0:
        print(f"Error running benchmark {dim}x{dim}:\n{proc.stderr}")
        return
        
    bend_elapsed_ms = (t1 - t0) * 1000.0
    
    # NumPy reference time
    t_np0 = time.perf_counter()
    for _ in range(100):
        _ = A @ x
    t_np1 = time.perf_counter()
    np_elapsed_us = (t_np1 - t_np0) * 10.0 # average per call in microseconds
    
    ops = 2 * dim * dim # FLOPs for matrix-vector product
    mflops = (ops / ((t1 - t0) * 1e6)) if (t1 - t0) > 0 else 0
    
    print(f"Matrix [{dim:3d} x {dim:3d}] | Total FLOPs: {ops:8d} | Bend Wall Time: {bend_elapsed_ms:6.2f} ms | Throughput: {mflops:7.2f} MFLOPs")

def main():
    print("=" * 75)
    print("      Bend GPU/Parallel Tree Matrix Multiplication Benchmark")
    print("=" * 75)
    for dim in [8, 16, 32, 64, 128]:
        benchmark_dimension(dim)
    print("=" * 75)

if __name__ == '__main__':
    main()
