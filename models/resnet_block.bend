#!/usr/bin/env bend
# Auto-generated Bend neural network compiled from ONNX
# Model: main_graph
import Base

# === Bend Matrix & Tensor Core Primitives ===
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

# === Model Parameters (Trained Weights & Biases) ===

def param_conv1_weight_f0() -> Vec:
  VCon{0.1802035, VCon{0.1956347, VCon{F32.neg(0.05521856), VCon{0.2165188, VCon{F32.neg(0.05164321), VCon{0.04756252, VCon{F32.neg(0.1147528), VCon{0.1384238, VCon{0.2077817, VCon{F32.neg(0.1729178), VCon{0.2048715, VCon{0.04411387, VCon{0.1741389, VCon{0.03192129, VCon{0.1136528, VCon{F32.neg(0.03327904), VCon{0.1816995, VCon{0.03483889, VNil{}}}}}}}}}}}}}}}}}}}

def param_conv1_weight_f1() -> Vec:
  VCon{F32.neg(0.1100352), VCon{0.06008011, VCon{F32.neg(0.1085967), VCon{F32.neg(0.02764148), VCon{F32.neg(0.09573244), VCon{0.156358, VCon{F32.neg(0.1860563), VCon{F32.neg(0.1086612), VCon{F32.neg(0.06655636), VCon{F32.neg(0.1417213), VCon{0.02224631, VCon{F32.neg(0.2327982), VCon{0.2128648, VCon{F32.neg(0.2002216), VCon{0.1819686, VCon{0.03922537, VCon{F32.neg(0.07653359), VCon{0.1456522, VNil{}}}}}}}}}}}}}}}}}}}

def param_conv1_bias() -> Vec:
  VCon{0.03673438, VCon{0.1904388, VNil{}}}

def param_conv2_weight_f0() -> Vec:
  VCon{0.0257669, VCon{F32.neg(0.07433417), VCon{0.06332935, VCon{F32.neg(0.06391761), VCon{0.09919707, VCon{0.2104402, VCon{0.1362499, VCon{F32.neg(0.1030426), VCon{0.1360626, VCon{0.04217332, VCon{0.1196979, VCon{F32.neg(0.1436617), VCon{F32.neg(0.2333237), VCon{F32.neg(0.09106605), VCon{F32.neg(0.180789), VCon{0.1934029, VCon{0.06788964, VCon{0.09763107, VNil{}}}}}}}}}}}}}}}}}}}

def param_conv2_weight_f1() -> Vec:
  VCon{0.07454348, VCon{F32.neg(0.00410027), VCon{0.1844625, VCon{F32.neg(0.1674697), VCon{0.01484069, VCon{F32.neg(0.1608763), VCon{0.07267926, VCon{F32.neg(0.08117168), VCon{0.07222301, VCon{F32.neg(0.04910656), VCon{0.1954895, VCon{F32.neg(0.1397012), VCon{F32.neg(0.1405724), VCon{F32.neg(0.1405808), VCon{0.2120007, VCon{0.07854804, VCon{0.2268047, VCon{F32.neg(0.1945195), VNil{}}}}}}}}}}}}}}}}}}}

def param_conv2_bias() -> Vec:
  VCon{F32.neg(0.2337874), VCon{F32.neg(0.1844049), VNil{}}}

# Patch extractor for Conv node 0
def extract_patches_0(+img: Vec) -> PatchTree:
  PTNode{PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}}

# Patch extractor for Conv node 2
def extract_patches_2(+img: Vec) -> PatchTree:
  PTNode{PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}}

# === Forward Inference Computation Graph ===
def forward(+input: Vec) -> Vec:
  +patches_0 : PatchTree = extract_patches_0(input)
  c_0_0 : Vec = conv_channel_patches(patches_0, param_conv1_weight_f0(), 0.03673438)
  c_0_1 : Vec = conv_channel_patches(patches_0, param_conv1_weight_f1(), 0.1904388)
  t0 : Vec = vec_concat(c_0_0, c_0_1)
  +t1 : Vec = vec_relu(t0)
  +patches_2 : PatchTree = extract_patches_2(t1)
  c_2_0 : Vec = conv_channel_patches(patches_2, param_conv2_weight_f0(), F32.neg(0.2337874))
  c_2_1 : Vec = conv_channel_patches(patches_2, param_conv2_weight_f1(), F32.neg(0.1844049))
  t2 : Vec = vec_concat(c_2_0, c_2_1)
  t3 : Vec = vec_add(t2, input)
  t4 : Vec = vec_relu(t3)
  t4

# === Main Entrypoint & GPU Parallel Verification ===
def main() -> IO(Unit):
  do IO<Unit>:
    test_x : Vec = VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
    
    # Run forward inference with parallel GPU tree dispatch:
    +out : Vec = forward!(test_x)
    
    predicted_class : U32 = vec_argmax(out)
    
    IO.print("BEND_ONNX_OUTPUT: " ++ vec_show(out))
    IO.print("BEND_PREDICTED_CLASS: " ++ U32.show(predicted_class))
