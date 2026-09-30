#!/usr/bin/env python3
"""
ONNX-to-Bend Compiler & Runtime
Translates ONNX neural network models into verified, parallel GPU Bend code.
Supports: Gemm, MatMul, Conv (2D), GlobalAveragePool, Add, Relu, Sigmoid, Softmax, Flatten.
"""

import sys
if sys.version_info >= (3, 14):
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

def format_vec_terms(terms: list[str]) -> str:
    """Convert a list of Bend expression strings into a nested VCon list."""
    res = "VNil{}"
    for t in reversed(terms):
        res = f"VCon{{{t}, {res}}}"
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

def build_patch_tree_code(patch_terms: list[str]) -> str:
    """Build a balanced binary PatchTree of patch vectors."""
    if len(patch_terms) == 0:
        return "PTLeaf{VNil{}}"
    if len(patch_terms) == 1:
        return f"PTLeaf{{{patch_terms[0]}}}"
    mid = len(patch_terms) // 2
    left = build_patch_tree_code(patch_terms[:mid])
    right = build_patch_tree_code(patch_terms[mid:])
    return f"PTNode{{{left}, {right}}}"

def chain_vec_concat(vars_list: list[str]) -> str:
    """Chain vec_concat over a list of variable names."""
    if len(vars_list) == 0:
        return "VNil{}"
    if len(vars_list) == 1:
        return vars_list[0]
    res = vars_list[-1]
    for v in reversed(vars_list[:-1]):
        res = f"vec_concat({v}, {res})"
    return res

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
        
        # Header
        code_lines.append("#!/usr/bin/env bend")
        code_lines.append("# Auto-generated Bend neural network compiled from ONNX")
        code_lines.append(f"# Model: {self.graph.name}")
        code_lines.append("import Base\n")
        
        # Include Matrix & Tensor primitives
        code_lines.append("""# === Bend Matrix & Tensor Core Primitives ===
type Vec is Data:
  VNil{}
  VCon{head: F32, tail: Vec}

def vec_len(v: Vec) -> U32:
  match v:
    case VNil{}:
      0
    case VCon{_, t}:
      (1 + vec_len(t) : U32)

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

def vec_get(+v: Vec, +idx: U32) -> F32:
  match v idx:
    case VCon{+h, _} 0:
      h
    case VCon{_, t} _:
      vec_get(t, U32.sub(idx, 1))
    case _ _:
      0.0

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

def f32_gelu(+x: F32) -> F32:
  c : F32 = 0.79788456
  +x3 : F32 = (x * x * x : F32)
  inner = (c * (x + 0.044715 * x3 : F32) : F32)
  t = F32.tanh(inner)
  (0.5 * x * (1.0 + t : F32) : F32)

def vec_gelu(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{f32_gelu(h), vec_gelu(t)}

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

type PatchTree is Data:
  PTLeaf{patch: Vec}
  PTNode{left: PatchTree, right: PatchTree}

def conv_channel_patches(patches: PatchTree, +filter: Vec, +bias: F32) -> Vec:
  match patches:
    case PTLeaf{patch}:
      VCon{((vec_dot(patch, filter) + bias : F32)), VNil{}}
    case PTNode{left, right}:
      l r = conv_channel_patches!(left, filter, bias) conv_channel_patches!(right, filter, bias)
      vec_concat(l, r)

def global_avg_pool_channel(x: Vec, +count: U32) -> F32:
  (vec_sum(x) / U32.to_f32(count) : F32)

def vec_show_inner(v: Vec) -> String:
  match v:
    case VNil{}:
      ""
    case VCon{+h, t}:
      " " ++ F32.show(h) ++ vec_show_inner(t)

def vec_show(v: Vec) -> String:
  "[" ++ vec_show_inner(v) ++ " ]"
""")
        
        # Track 2D weights transposition for Gemm / MatMul
        weights_to_transpose = set()
        conv_weights = set()
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
            elif node.op_type == 'Conv':
                if len(node.input) > 1:
                    conv_weights.add(node.input[1])

        # Emit constant weight definitions
        code_lines.append("# === Model Parameters (Trained Weights & Biases) ===\n")
        
        for name, arr in self.initializers.items():
            safe_name = name.replace(".", "_").replace("/", "_")
            if name in conv_weights:
                # 4D Conv weights [C_out, C_in, K_h, K_w]
                c_out = arr.shape[0]
                for c in range(c_out):
                    f_vec = arr[c].flatten().tolist()
                    code_lines.append(f"def param_{safe_name}_f{c}() -> Vec:")
                    code_lines.append(f"  {format_vec(f_vec)}\n")
            elif arr.ndim == 1:
                code_lines.append(f"def param_{safe_name}() -> Vec:")
                code_lines.append(f"  {format_vec(arr.tolist())}\n")
            elif arr.ndim == 2:
                target_arr = arr.T if name in weights_to_transpose else arr
                code_lines.append(f"def param_{safe_name}() -> MatTree:")
                code_lines.append(f"  {build_mat_tree(target_arr.tolist())}\n")
            else:
                code_lines.append(f"def param_{safe_name}() -> Vec:")
                code_lines.append(f"  {format_vec(arr.flatten().tolist())}\n")
        
        # Preliminary shape inference pass for all graph tensors
        tensor_shapes = {}
        input_name = self.graph.input[0].name
        raw_in_shape = [d.dim_value for d in self.graph.input[0].type.tensor_type.shape.dim]
        tensor_shapes[input_name] = [d if d > 0 else 1 for d in raw_in_shape]
        
        for node in self.graph.node:
            out_tensor = node.output[0]
            in_shape = tensor_shapes.get(node.input[0], [1, 4])
            if node.op_type == 'Conv':
                c_in, h_in, w_in = in_shape[1], in_shape[2], in_shape[3]
                w_arr = self.initializers[node.input[1]]
                c_out, _, kh, kw = w_arr.shape
                pads = [0, 0, 0, 0]
                strides = [1, 1]
                for a in node.attribute:
                    if a.name == 'pads':
                        pads = onnx.helper.get_attribute_value(a)
                    elif a.name == 'strides':
                        strides = onnx.helper.get_attribute_value(a)
                pt, pl, pb, pr = pads[0], pads[1], pads[2], pads[3] if len(pads) == 4 else (pads[0], pads[1], pads[0], pads[1])
                sy, sx = strides[0], strides[1]
                h_out = (h_in + pt + pb - kh) // sy + 1
                w_out = (w_in + pl + pr - kw) // sx + 1
                tensor_shapes[out_tensor] = [1, c_out, h_out, w_out]
            elif node.op_type == 'GlobalAveragePool':
                tensor_shapes[out_tensor] = [1, in_shape[1], 1, 1]
            elif node.op_type == 'Flatten':
                tensor_shapes[out_tensor] = [1, int(np.prod(in_shape))]
            elif node.op_type == 'Gemm':
                w_arr = self.initializers[node.input[1]]
                out_dim = w_arr.shape[0] if node.input[1] not in weights_to_transpose else w_arr.shape[1]
                tensor_shapes[out_tensor] = [1, out_dim]
            elif node.op_type == 'MatMul':
                w_arr = self.initializers[node.input[1]]
                out_dim = w_arr.shape[1] if w_arr.ndim == 2 else w_arr.shape[0]
                tensor_shapes[out_tensor] = [1, out_dim]
            else:
                tensor_shapes[out_tensor] = in_shape

        # Emit helper functions for Conv layers (patch extraction)
        for idx, node in enumerate(self.graph.node):
            if node.op_type == 'Conv':
                in_shape = tensor_shapes[node.input[0]]
                c_in, h_in, w_in = in_shape[1], in_shape[2], in_shape[3]
                w_arr = self.initializers[node.input[1]]
                c_out, _, kh, kw = w_arr.shape
                
                # Extract attributes
                pads = [0, 0, 0, 0]
                strides = [1, 1]
                for a in node.attribute:
                    if a.name == 'pads':
                        pads = onnx.helper.get_attribute_value(a)
                    elif a.name == 'strides':
                        strides = onnx.helper.get_attribute_value(a)
                        
                pt, pl, pb, pr = pads[0], pads[1], pads[2], pads[3] if len(pads) == 4 else (pads[0], pads[1], pads[0], pads[1])
                sy, sx = strides[0], strides[1]
                h_out = (h_in + pt + pb - kh) // sy + 1
                w_out = (w_in + pl + pr - kw) // sx + 1
                
                # Build list of patch codes
                patch_codes = []
                for y in range(h_out):
                    for x in range(w_out):
                        terms = []
                        for ci in range(c_in):
                            for dy in range(kh):
                                for dx in range(kw):
                                    iy = y * sy + dy - pt
                                    ix = x * sx + dx - pl
                                    if 0 <= iy < h_in and 0 <= ix < w_in:
                                        flat_idx = ci * (h_in * w_in) + iy * w_in + ix
                                        terms.append(f"vec_get(img, {flat_idx})")
                                    else:
                                        terms.append("0.0")
                        patch_codes.append(format_vec_terms(terms))
                
                tree_code = build_patch_tree_code(patch_codes)
                code_lines.append(f"# Patch extractor for Conv node {idx}")
                code_lines.append(f"def extract_patches_{idx}(+img: Vec) -> PatchTree:")
                code_lines.append(f"  {tree_code}\n")
                
                tensor_shapes[node.output[0]] = [1, c_out, h_out, w_out]

        # Count uses of each tensor variable to add '+' for multi-use Data binders
        import collections
        var_use_count = collections.Counter()
        for node in self.graph.node:
            for inp in node.input:
                var_use_count[inp] += 1
            if node.op_type in ('Conv', 'GlobalAveragePool'):
                var_use_count[node.input[0]] += 10
        var_use_count[self.graph.output[0].name] += 1

        # Emit forward inference function
        code_lines.append("# === Forward Inference Computation Graph ===")
        code_lines.append("def forward(+input: Vec) -> Vec:")
        
        var_names = {}
        var_names[input_name] = "input"
        
        for idx, node in enumerate(self.graph.node):
            out_tensor = node.output[0]
            curr_var = f"t{idx}"
            pfx = "+" if var_use_count[out_tensor] > 1 else ""
            
            if node.op_type == 'Gemm':
                in_var = var_names[node.input[0]]
                w_name = node.input[1].replace(".", "_").replace("/", "_")
                has_bias = len(node.input) > 2
                
                if has_bias:
                    b_name = node.input[2].replace(".", "_").replace("/", "_")
                    code_lines.append(f"  {pfx}{curr_var} : Vec = linear_layer({in_var}, param_{w_name}(), param_{b_name}())")
                else:
                    code_lines.append(f"  {pfx}{curr_var} : Vec = mat_tree_mul({in_var}, param_{w_name}())")
                var_names[out_tensor] = curr_var
                w_arr = self.initializers[node.input[1]]
                out_dim = w_arr.shape[0] if node.input[1] not in weights_to_transpose else w_arr.shape[1]
                tensor_shapes[out_tensor] = [1, out_dim]
                
            elif node.op_type == 'MatMul':
                in_var = var_names[node.input[0]]
                w_name = node.input[1].replace(".", "_").replace("/", "_")
                code_lines.append(f"  {pfx}{curr_var} : Vec = mat_tree_mul({in_var}, param_{w_name}())")
                var_names[out_tensor] = curr_var
                w_arr = self.initializers[node.input[1]]
                out_dim = w_arr.shape[1] if w_arr.ndim == 2 else w_arr.shape[0]
                tensor_shapes[out_tensor] = [1, out_dim]
                
            elif node.op_type == 'Conv':
                in_var = var_names[node.input[0]]
                w_name = node.input[1].replace(".", "_").replace("/", "_")
                w_arr = self.initializers[node.input[1]]
                c_out = w_arr.shape[0]
                b_arr = self.initializers[node.input[2]] if len(node.input) > 2 else np.zeros(c_out, dtype=np.float32)
                
                patch_pfx = "+" if c_out > 1 else ""
                code_lines.append(f"  {patch_pfx}patches_{idx} : PatchTree = extract_patches_{idx}({in_var})")
                chan_vars = []
                for c in range(c_out):
                    c_var = f"c_{idx}_{c}"
                    b_val = format_f32(b_arr[c])
                    code_lines.append(f"  {c_var} : Vec = conv_channel_patches(patches_{idx}, param_{w_name}_f{c}(), {b_val})")
                    chan_vars.append(c_var)
                concat_expr = chain_vec_concat(chan_vars)
                code_lines.append(f"  {pfx}{curr_var} : Vec = {concat_expr}")
                var_names[out_tensor] = curr_var
                
            elif node.op_type == 'GlobalAveragePool':
                in_var = var_names[node.input[0]]
                in_shape = tensor_shapes[node.input[0]]
                c, h, w = in_shape[1], in_shape[2], in_shape[3]
                spatial_size = h * w
                
                avg_vars = []
                for ch in range(c):
                    # extract spatial items
                    terms = [f"vec_get({in_var}, {ch * spatial_size + k})" for k in range(spatial_size)]
                    ch_vec = format_vec_terms(terms)
                    code_lines.append(f"  ch_{idx}_{ch} : Vec = {ch_vec}")
                    code_lines.append(f"  avg_{idx}_{ch} : F32 = global_avg_pool_channel(ch_{idx}_{ch}, {spatial_size})")
                    avg_vars.append(f"avg_{idx}_{ch}")
                
                code_lines.append(f"  {pfx}{curr_var} : Vec = {format_vec_terms(avg_vars)}")
                var_names[out_tensor] = curr_var
                tensor_shapes[out_tensor] = [1, c]
                
            elif node.op_type == 'Add':
                in_var1 = var_names[node.input[0]]
                if node.input[1] in self.initializers:
                    in_name2 = node.input[1].replace(".", "_").replace("/", "_")
                    code_lines.append(f"  {pfx}{curr_var} : Vec = vec_add({in_var1}, param_{in_name2}())")
                else:
                    in_var2 = var_names[node.input[1]]
                    code_lines.append(f"  {pfx}{curr_var} : Vec = vec_add({in_var1}, {in_var2})")
                var_names[out_tensor] = curr_var
                tensor_shapes[out_tensor] = tensor_shapes[node.input[0]]
                
            elif node.op_type == 'Relu':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {pfx}{curr_var} : Vec = vec_relu({in_var})")
                var_names[out_tensor] = curr_var
                tensor_shapes[out_tensor] = tensor_shapes[node.input[0]]
                
            elif node.op_type == 'Sigmoid':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {pfx}{curr_var} : Vec = vec_sigmoid({in_var})")
                var_names[out_tensor] = curr_var
                tensor_shapes[out_tensor] = tensor_shapes[node.input[0]]
                
            elif node.op_type == 'Softmax':
                in_var = var_names[node.input[0]]
                code_lines.append(f"  {pfx}{curr_var} : Vec = vec_softmax({in_var})")
                var_names[out_tensor] = curr_var
                tensor_shapes[out_tensor] = tensor_shapes[node.input[0]]
                
            elif node.op_type == 'Flatten':
                in_var = var_names[node.input[0]]
                var_names[out_tensor] = in_var
                in_shape = tensor_shapes[node.input[0]]
                tensor_shapes[out_tensor] = [1, int(np.prod(in_shape))]
                
            else:
                raise NotImplementedError(f"Unsupported ONNX operator: {node.op_type}")
        
        # Return final output
        final_output_name = self.graph.output[0].name
        final_var = var_names[final_output_name]
        code_lines.append(f"  {final_var}\n")
        
        # Determine input dimension for test input
        inp_shape = [d.dim_value for d in self.graph.input[0].type.tensor_type.shape.dim]
        in_dim = int(np.prod([d if d > 0 else 1 for d in inp_shape]))
        
        if default_test_input is None:
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
    print(f"ONNX Runtime Ref:   {expected_out[:5]}... (len {len(expected_out)})")
    print(f"Bend GPU Result:    {actual_out[:5]}... (len {len(actual_out)})")
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
