#!/usr/bin/env python3
"""
CLI Tool for Running Neural Network Inference via Bend GPU/Parallel Runtime.

Usage:
  python3 run_inference.py models/mlp_classifier.onnx --input "0.5, 0.2, -0.1, 0.4"
  python3 run_inference.py models/mlp_classifier.bend
"""

import sys
for p in [
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
    "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/site-packages",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import argparse
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.onnx_compiler import OnnxToBendCompiler, run_bend, parse_bend_vector

BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

def main():
    parser = argparse.ArgumentParser(description="Run ONNX/Bend Neural Network Inference")
    parser.add_argument("model", help="Path to .onnx or .bend model file")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Comma-separated input vector, e.g. '0.5, -0.2, 1.0, 0.3'")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Compare with ONNX Runtime reference if given .onnx file")
    
    args = parser.parse_args()
    
    if args.model.endswith(".onnx"):
        compiler = OnnxToBendCompiler(args.model)
        
        # Parse or default test input
        inp_shape = [d.dim_value if d.dim_value > 0 else 1 for d in compiler.graph.input[0].type.tensor_type.shape.dim]
        in_dim = inp_shape[-1]
        
        if args.input:
            input_vals = [float(x.strip()) for x in args.input.split(",")]
            if len(input_vals) != in_dim:
                print(f"Warning: input vector has length {len(input_vals)}, expected {in_dim}")
        else:
            print(f"No --input provided. Using default sample input of length {in_dim}...")
            input_vals = [0.5] * in_dim
            
        temp_bend = f"/tmp/run_{os.path.basename(args.model)}.bend"
        bend_code = compiler.compile(default_test_input=input_vals)
        with open(temp_bend, "w") as f:
            f.write(bend_code)
        target_bend_file = temp_bend
        
    elif args.model.endswith(".bend"):
        target_bend_file = args.model
    else:
        print("Error: model must be .onnx or .bend file")
        sys.exit(1)
        
    print(f"\n🚀 Running inference using Bend on {target_bend_file}...")
    stdout = run_bend(target_bend_file)
    output_vec = parse_bend_vector(stdout)
    
    pred_class = None
    for line in stdout.splitlines():
        if line.startswith("BEND_PREDICTED_CLASS:"):
            pred_class = int(line.split(":")[1].strip())
            
    print("\n" + "=" * 50)
    print("           BEND INFERENCE RESULTS")
    print("=" * 50)
    print(f"Output Vector:    {output_vec}")
    if pred_class is not None:
        print(f"Predicted Class:  {pred_class}")
    print("=" * 50 + "\n")
    
    if args.compare and args.model.endswith(".onnx"):
        from src.onnx_compiler import run_onnxruntime
        arr_input = np.array(input_vals, dtype=np.float32).reshape(inp_shape)
        ort_out = run_onnxruntime(args.model, arr_input).flatten()
        print(f"ONNX Runtime Ref: {ort_out.tolist()}")
        err = np.max(np.abs(np.array(output_vec) - ort_out))
        print(f"Max Absolute Err: {err:.8e}")

if __name__ == '__main__':
    main()
