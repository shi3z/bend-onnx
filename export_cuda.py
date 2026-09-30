#!/usr/bin/env python3
"""
Exports a Bend neural network model into standalone unified C/CUDA source code.
On an NVIDIA GPU system, the generated .c file compiles directly via:
  clang -O3 model.c -o model
or via Bend's native GPU runner.
"""

import sys
import os
import subprocess

BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

def export_cuda(bend_path: str, output_c_path: str):
    print(f"Translating Bend model '{bend_path}' into unified C/CUDA...")
    proc = subprocess.run([BEND_BIN, bend_path, "-o", output_c_path], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error compiling to C/CUDA:\n{proc.stderr}")
        return False
        
    print(f"✅ Generated C/CUDA source at: {output_c_path}")
    print("\nCUDA & GPU Integration details in emitted code:")
    with open(output_c_path) as f:
        lines = f.readlines()
        cuda_lines = [line.strip() for line in lines if any(k in line.lower() for k in ["cuda", "gpu", "rtc", "metal"])][:10]
        for l in cuda_lines:
            print(f"  {l}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 export_cuda.py <model.bend> [output.c]")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else in_file.replace(".bend", ".c")
    export_cuda(in_file, out_file)
