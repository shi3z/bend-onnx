#!/usr/bin/env bend
# Auto-generated Bend neural network compiled from ONNX
# Model: pure_matmul
import Base

# === Bend Matrix & Tensor Core Primitives ===
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

# === Model Parameters (Trained Weights & Biases) ===

def param_B() -> MatTree:
  MTNode{
    MTLeaf{VCon{1.0, VCon{3.0, VCon{5.0, VNil{}}}}},
    MTLeaf{VCon{2.0, VCon{4.0, VCon{6.0, VNil{}}}}}
  }

# === Forward Inference Computation Graph ===
def forward(+input: Vec) -> Vec:
  t0 : Vec = mat_tree_mul(input, param_B())
  t0

# === Main Entrypoint & GPU Parallel Verification ===
def main() -> IO(Unit):
  do IO<Unit>:
    test_x : Vec = VCon{0.5, VCon{0.5, VCon{0.5, VNil{}}}}
    
    # Run forward inference with parallel GPU tree dispatch:
    +out : Vec = forward!(test_x)
    
    predicted_class : U32 = vec_argmax(out)
    
    IO.print("BEND_ONNX_OUTPUT: " ++ vec_show(out))
    IO.print("BEND_PREDICTED_CLASS: " ++ U32.show(predicted_class))
