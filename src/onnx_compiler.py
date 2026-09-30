#!/usr/bin/env python3
"""
ONNX-to-Bend Compiler & Runtime
Translates ONNX neural network models into verified, parallel GPU Bend code.
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
import onnx
from onnx import numpy_helper

BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

def format_f32(v: float) -> str:
    """Format a float value for Bend syntax (handling negative numbers safely)."""
    val = float(v)
    if val == 0.0:
        return "0.0"
    is_neg = val < 0.0
    abs_val = abs(val)
    s = f"{abs_val:.7g}"
    if "." not in s and "e" not in s:
        s += ".0"
    if is_neg:
        return f"F32.neg({s})"
    return s

def format_vec(vals) -> str:
    """Convert a 1D iterable of floats into a Bend Vec."""
    res = "VNil{}"
    for v in reversed(vals):
        res = f"VCon{{{format_f32(v)}, {res}}}"
    return res

def build_mat_tree(rows) -> str:
    """Build a balanced binary MatTree from a 2D matrix (list of rows)."""
    if len(rows) == 0:
        return "MTLeaf{VNil{}}"
    if len(rows) == 1:
        return f"MTLeaf{{{format_vec(rows[0])}}}"
    
    mid = len(rows) // 2
    left = build_mat_tree(rows[:mid])
    right = build_mat_tree(rows[mid:])
    return f"MTNode{{\n    {left},\n    {right}\n  }}"

class OnnxToBendCompiler:
    def __init__(self, onnx_model_path: str):
        self.model_path = onnx_model_path
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.initializers = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = numpy_helper.to_array(init)

    def inspect(self):
        """Prints a human-readable summary of the ONNX graph."""
        print(f"=== ONNX Model Summary: {os.path.basename(self.model_path)} ===")
        print(f"IR Version: {self.model.ir_version}, Producer: {self.model.producer_name}")
        
        print("\n--- Inputs ---")
        for inp in self.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: shape {shape}")
            
        print("\n--- Outputs ---")
        for out in self.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: shape {shape}")
            
        print("\n--- Initializers (Weights & Biases) ---")
        total_params = 0
        for name, arr in self.initializers.items():
            total_params += arr.size
            print(f"  {name}: shape {arr.shape}, dtype {arr.dtype}, params: {arr.size}")
        print(f"Total Parameters: {total_params}")
        
        print("\n--- Computation Graph Nodes ---")
        for i, node in enumerate(self.graph.node):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            print(f"  [{i}] {node.op_type}: in={node.input} -> out={node.output} attrs={attrs}")

    def compile(self, default_test_input=None) -> str:
        """Transpiles the ONNX graph into complete, verified Bend code."""
        code_lines = []
        
        # Header and module imports
        code_lines.append("#!/usr/bin/env bend")
        code_lines.append("# Auto-generated Bend neural network compiled from ONNX")
        code_lines.append(f"# Model: {self.graph.name}")
        code_lines.append("import Base\n")
        
        # Include Matrix & Tensor primitives
        code_lines.append("""# === Bend Matrix & Tensor Core Primitives ===
type Vec is Data:
  VNil{}
  VCon{head: F32, tail: Vec}

def vec_dot_acc(x: Vec, y: Vec, acc: F32) -> F32:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      vec_dot_acc(xt, yt, (acc + xh * yh : F32))
    case _ _:
      acc

def vec_dot(x: Vec, y: Vec) -> F32:
  vec_dot_acc(x, y, 0.0)

def vec_add(x: Vec, y: Vec) -> Vec:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      VCon{(xh + yh : F32), vec_add(xt, yt)}
    case _ _:
      VNil{}

def vec_concat(x: Vec, y: Vec) -> Vec:
  match x:
    case VNil{}:
      y
    case VCon{h, t}:
      VCon{h, vec_concat(t, y)}

def vec_relu(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{F32.max(0.0, h), vec_relu(t)}

def f32_sigmoid(x: F32) -> F32:
  neg_x = F32.neg(x)
  e = F32.exp(neg_x)
  (1.0 / (1.0 + e : F32) : F32)

def vec_sigmoid(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{f32_sigmoid(h), vec_sigmoid(t)}

def vec_max_acc(v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vec_max_acc(t, F32.max(acc, h))

def vec_max(v: Vec) -> F32:
  match v:
    case VNil{}:
      0.0
    case VCon{+h, t}:
      vec_max_acc(t, h)

def vec_exp_shifted(+shift: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{F32.exp((h - shift : F32)), vec_exp_shifted(shift, t)}

def vec_sum_acc(v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vec_sum_acc(t, (acc + h : F32))

def vec_sum(v: Vec) -> F32:
  vec_sum_acc(v, 0.0)

def vec_scale_div(+denom: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(h / denom : F32), vec_scale_div(denom, t)}

def vec_softmax(+v: Vec) -> Vec:
  m : F32 = vec_max(v)
  +exps : Vec = vec_exp_shifted(m, v)
  s : F32 = vec_sum(exps)
  vec_scale_div(s, exps)

def vec_argmax_go(v: Vec, +best_idx: U32, +best_val: F32, +curr_idx: U32) -> U32:
  match v:
    case VNil{}:
      best_idx
    case VCon{+h, t}:
      +is_better = F32.is_gt(h, best_val)
      new_idx = Bool.pick(U32, is_better, curr_idx, best_idx)
      new_val = Bool.pick(F32, is_better, h, best_val)
      vec_argmax_go(t, new_idx, new_val, (curr_idx + 1 : U32))

def vec_argmax(v: Vec) -> U32:
  match v:
    case VNil{}:
      0
    case VCon{+h, t}:
      vec_argmax_go(t, 0, h, 1)

type MatTree is Data:
  MTLeaf{vec: Vec}
  MTNode{left: MatTree, right: MatTree}

def mat_tree_mul(+x: Vec, m: MatTree) -> Vec:
  match m:
    case MTLeaf{row}:
      VCon{vec_dot(row, x), VNil{}}
    case MTNode{left, right}:
      l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
      vec_concat(l, r)

def linear_layer(+x: Vec, w: MatTree, b: Vec) -> Vec:
  raw : Vec = mat_tree_mul(x, w)
  vec_add(raw, b)

def vec_show_inner(v: Vec) -> String:
  match v:
    case VNil{}:
      ""
    case VCon{+h, t}:
      " " ++ F32.show(h) ++ vec_show_inner(t)

def vec_show(v: Vec) -> String:
  "[" ++ vec_show_inner(v) ++ " ]"
""")
        
        # Emit constant weight definitions
        code_lines.append("# === Model Parameters (Trained Weights & Biases) ===\n")
        
        # Determine which 2D initializers need transposition
        # In our row-wise mat_tree_mul, weights are stored as (out_features, in_features)
        weights_to_transpose = set()
        for node in self.graph.node:
            if node.op_type == 'MatMul':
                if len(node.input) > 1 and node.input[1] in self.initializers:
                    weights_to_transpose.add(node.input[1])
            elif node.op_type == 'Gemm':
                trans_b = 0
                for a in node.attribute:
                    if a.name == 'transB':
                        trans_b = onnx.helper.get_attribute_value(a)
                if trans_b == 0 and len(node.input) > 1 and node.input[1] in self.initializers:
                    weights_to_transpose.add(node.input[1])

        for name, arr in self.initializers.items():
            safe_name = name.replace(".", "_").replace("/", "_")
            if arr.ndim == 1:
                code_lines.append(f"def param_{safe_name}() -> Vec:")
                code_lines.append(f"  {format_vec(arr.tolist())}\n")
            elif arr.ndim == 2:
                target_arr = arr.T if name in weights_to_transpose else arr
                code_lines.append(f"def param_{safe_name}() -> MatTree:")
                code_lines.append(f"  {build_mat_tree(target_arr.tolist())}\n")
        
        # Emit forward inference function
        code_lines.append("# === Forward Inference Computation Graph ===")
        code_lines.append("def forward(+input: Vec) -> Vec:")
        
        # Trace node dependencies
        var_names = {}
        # Input tensor
        input_name = self.graph.input[0].name
        var_names[input_name] = "input"
        
        for idx, node in enumerate(self.graph.node):
            out_tensor = node.output[0]
            curr_var = f"t{idx}"
            
            if node.op_type == 'Gemm':
                in_var = var_names[node.input[0]]
                w_name = node.input[1].replace(".", "_").replace("/", "_")
                has_bias = len(node.input) > 2
                
                if has_bias:
                    b_name = node.input[2].replace(".", "_").replace("/", "_")
                    code_lines.append(f"  {curr_var} : Vec = linear_layer({in_var}, param_{w_name}(), param_{b_name}())")
                else:
                    code_lines.append(f"  {curr_var} : Vec = mat_tree_mul({in_var}, param_{w_name}())")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'MatMul':
                in_var = var_names[node.input[0]]
                w_name = node.input[1].replace(".", "_").replace("/", "_")
                code_lines.append(f"  {curr_var} : Vec = mat_tree_mul({in_var}, param_{w_name}())")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'Add':
                in_var1 = var_names[node.input[0]]
                in_name2 = node.input[1].replace(".", "_").replace("/", "_")
                if in_name2 in self.initializers:
                    code_lines.append(f"  {curr_var} : Vec = vec_add({in_var1}, param_{in_name2}())")
                else:
                    in_var2 = var_names[node.input[1]]
                    code_lines.append(f"  {curr_var} : Vec = vec_add({in_var1}, {in_var2})")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'Relu':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {curr_var} : Vec = vec_relu({in_var})")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'Sigmoid':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {curr_var} : Vec = vec_sigmoid({in_var})")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'Softmax':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {curr_var} : Vec = vec_softmax({in_var})")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'Flatten':
                in_var = var_names[node.input[0]]
                var_names[out_tensor] = in_var
                
            else:
                raise NotImplementedError(f"Unsupported ONNX operator: {node.op_type}")
        
        # Return final output
        final_output_name = self.graph.output[0].name
        final_var = var_names[final_output_name]
        code_lines.append(f"  {final_var}\n")
        
        # Determine input dimension for test input
        inp_shape = [d.dim_value for d in self.graph.input[0].type.tensor_type.shape.dim]
        in_dim = inp_shape[-1] if inp_shape else 4
        
        if default_test_input is None:
            # Generate deterministic test input: [0.5, 0.5, ...]
            test_vec_code = format_vec([0.5] * in_dim)
        else:
            test_vec_code = format_vec(default_test_input)
            
        code_lines.append(f"""# === Main Entrypoint & GPU Parallel Verification ===
def main() -> IO(Unit):
  do IO<Unit>:
    test_x : Vec = {test_vec_code}
    
    # Run forward inference with parallel GPU tree dispatch:
    +out : Vec = forward!(test_x)
    
    predicted_class : U32 = vec_argmax(out)
    
    IO.print("BEND_ONNX_OUTPUT: " ++ vec_show(out))
    IO.print("BEND_PREDICTED_CLASS: " ++ U32.show(predicted_class))
""")
        
        return "\n".join(code_lines)

def run_bend(code_path: str):
    """Executes a Bend program and captures its output."""
    proc = subprocess.run([BEND_BIN, code_path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Bend execution failed:\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout

def run_onnxruntime(onnx_path: str, input_data: np.ndarray) -> np.ndarray:
    """Runs reference inference using ONNX Runtime."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    session = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data.astype(np.float32)})
    return outputs[0]

def parse_bend_vector(stdout: str) -> list[float]:
    """Parses output vector from Bend stdout."""
    for line in stdout.splitlines():
        if line.startswith("BEND_ONNX_OUTPUT:"):
            # Format: BEND_ONNX_OUTPUT: [ 0.1 0.2 ... ]
            content = line.split(":", 1)[1].strip()
            if content.startswith("[") and content.endswith("]"):
                inner = content[1:-1].strip()
                if not inner:
                    return []
                return [float(x) for x in inner.split()]
    raise ValueError(f"Could not find BEND_ONNX_OUTPUT in:\n{stdout}")

def verify(onnx_path: str, temp_bend_path: str = "/tmp/temp_model.bend"):
    """Validates Bend output against ONNX Runtime reference."""
    compiler = OnnxToBendCompiler(onnx_path)
    
    # Determine input shape
    inp_info = compiler.graph.input[0]
    shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp_info.type.tensor_type.shape.dim]
    np.random.seed(42)
    test_input = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
    
    # 1. Run ONNX Runtime
    expected_out = run_onnxruntime(onnx_path, test_input).flatten()
    
    # 2. Compile to Bend and run
    flat_input = test_input.flatten().tolist()
    bend_code = compiler.compile(default_test_input=flat_input)
    with open(temp_bend_path, "w") as f:
        f.write(bend_code)
        
    bend_out_str = run_bend(temp_bend_path)
    actual_out = np.array(parse_bend_vector(bend_out_str), dtype=np.float32)
    
    # 3. Compare outputs
    max_err = np.max(np.abs(expected_out - actual_out))
    mean_err = np.mean(np.abs(expected_out - actual_out))
    
    print(f"\n================ Verification for {os.path.basename(onnx_path)} ================")
    print(f"Input Shape:        {test_input.shape}")
    print(f"Output Shape:       {expected_out.shape}")
    print(f"ONNX Runtime Ref:   {expected_out}")
    print(f"Bend GPU Result:    {actual_out}")
    print(f"Max Absolute Error: {max_err:.8e}")
    print(f"Mean Abs Error:     {mean_err:.8e}")
    
    if max_err < 1e-4:
        print(">>> RESULT: PASSED (Identical within float32 tolerance) <<<")
        return True
    else:
        print(">>> RESULT: FAILED (Numerical divergence detected) <<<")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compile and run ONNX models on Bend GPU")
    parser.add_argument("model", help="Path to .onnx file")
    parser.add_argument("-o", "--output", help="Path to output .bend file")
    parser.add_argument("--inspect", action="store_true", help="Print model summary")
    parser.add_argument("--verify", action="store_true", help="Verify against ONNX Runtime")
    parser.add_argument("--run", action="store_true", help="Compile and execute model")
    
    args = parser.parse_args()
    
    compiler = OnnxToBendCompiler(args.model)
    
    if args.inspect:
        compiler.inspect()
        return
        
    if args.verify:
        verify(args.model)
        return
        
    out_path = args.output or args.model.replace(".onnx", ".bend")
    bend_code = compiler.compile()
    with open(out_path, "w") as f:
        f.write(bend_code)
    print(f"Compiled Bend model saved to: {out_path}")
    
    if args.run:
        print(f"\nRunning {out_path} with Bend...")
        out = run_bend(out_path)
        print(out)

if __name__ == '__main__':
    main()
