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

def param_conv_in_weight_f0() -> Vec:
  VCon{F32.neg(0.2917374), VCon{F32.neg(0.212246), VCon{0.333203, VCon{0.06295832, VCon{0.1027199, VCon{F32.neg(0.3108948), VCon{F32.neg(0.2189246), VCon{F32.neg(0.1109519), VCon{0.05212371, VNil{}}}}}}}}}}

def param_conv_in_weight_f1() -> Vec:
  VCon{F32.neg(0.2933071), VCon{F32.neg(0.1436244), VCon{F32.neg(0.1995562), VCon{0.0009237528, VCon{F32.neg(0.1240344), VCon{F32.neg(0.02309859), VCon{F32.neg(0.2258766), VCon{F32.neg(0.2287984), VCon{F32.neg(0.1944673), VNil{}}}}}}}}}}

def param_conv_in_bias() -> Vec:
  VCon{F32.neg(0.1140991), VCon{F32.neg(0.2630937), VNil{}}}

def param_res_block_conv1_weight_f0() -> Vec:
  VCon{0.1976292, VCon{F32.neg(0.04677842), VCon{0.2027975, VCon{0.0734406, VCon{F32.neg(0.199592), VCon{0.1631143, VCon{F32.neg(0.06485225), VCon{F32.neg(0.09035081), VCon{F32.neg(0.1956495), VCon{F32.neg(0.2343259), VCon{0.06743693, VCon{F32.neg(0.05148771), VCon{0.09176434, VCon{F32.neg(0.1934322), VCon{0.1749922, VCon{F32.neg(0.1730181), VCon{F32.neg(0.0406995), VCon{0.04923106, VNil{}}}}}}}}}}}}}}}}}}}

def param_res_block_conv1_weight_f1() -> Vec:
  VCon{0.1216817, VCon{0.1902849, VCon{0.2147152, VCon{F32.neg(0.1868935), VCon{0.05931854, VCon{F32.neg(0.1013817), VCon{F32.neg(0.02582939), VCon{F32.neg(0.1764208), VCon{0.2146914, VCon{F32.neg(0.1729938), VCon{0.1259714, VCon{0.08283511, VCon{0.07659285, VCon{F32.neg(0.1274315), VCon{0.2142419, VCon{0.05179567, VCon{0.03032075, VCon{F32.neg(0.2077138), VNil{}}}}}}}}}}}}}}}}}}}

def param_res_block_conv1_bias() -> Vec:
  VCon{0.0989451, VCon{F32.neg(0.03536019), VNil{}}}

def param_res_block_conv2_weight_f0() -> Vec:
  VCon{F32.neg(0.1079809), VCon{0.2024556, VCon{0.05254953, VCon{F32.neg(0.1304085), VCon{F32.neg(0.1192981), VCon{F32.neg(0.01125613), VCon{0.131607, VCon{F32.neg(0.0602299), VCon{F32.neg(0.1344858), VCon{F32.neg(0.08071456), VCon{F32.neg(0.1760872), VCon{0.08405908, VCon{0.182443, VCon{F32.neg(0.2218996), VCon{0.05474201, VCon{0.1217619, VCon{0.04273974, VCon{F32.neg(0.08393938), VNil{}}}}}}}}}}}}}}}}}}}

def param_res_block_conv2_weight_f1() -> Vec:
  VCon{0.1230229, VCon{0.1238646, VCon{0.0881355, VCon{F32.neg(0.04141793), VCon{F32.neg(0.06241425), VCon{0.02521564, VCon{F32.neg(0.04163777), VCon{F32.neg(0.07023952), VCon{0.1506625, VCon{0.2025624, VCon{F32.neg(0.0233339), VCon{F32.neg(0.052773), VCon{0.003439436, VCon{F32.neg(0.01407337), VCon{0.05666548, VCon{0.0660517, VCon{F32.neg(0.2140782), VCon{F32.neg(0.08698303), VNil{}}}}}}}}}}}}}}}}}}}

def param_res_block_conv2_bias() -> Vec:
  VCon{0.1984918, VCon{0.09181899, VNil{}}}

def param_fc_weight() -> MatTree:
  MTNode{
    MTLeaf{VCon{F32.neg(0.03516977), VCon{F32.neg(0.4263187), VNil{}}}},
    MTNode{
    MTLeaf{VCon{F32.neg(0.4326115), VCon{F32.neg(0.6334028), VNil{}}}},
    MTLeaf{VCon{F32.neg(0.2304902), VCon{0.2387929, VNil{}}}}
  }
  }

def param_fc_bias() -> Vec:
  VCon{0.4508666, VCon{0.3264693, VCon{F32.neg(0.6250428), VNil{}}}}

# Patch extractor for Conv node 0
def extract_patches_0(+img: Vec) -> PatchTree:
  PTNode{PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VNil{}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VNil{}}}}}}}}}}}}, PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VNil{}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VNil{}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VNil{}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VNil{}}}}}}}}}}}}}}, PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VNil{}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VNil{}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}

# Patch extractor for Conv node 2
def extract_patches_2(+img: Vec) -> PatchTree:
  PTNode{PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}}

# Patch extractor for Conv node 4
def extract_patches_4(+img: Vec) -> PatchTree:
  PTNode{PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 0), VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 16), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 1), VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 17), VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 2), VCon{vec_get(img, 3), VCon{0.0, VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 18), VCon{vec_get(img, 19), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 4), VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 20), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 5), VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{vec_get(img, 21), VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 6), VCon{vec_get(img, 7), VCon{0.0, VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{vec_get(img, 22), VCon{vec_get(img, 23), VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}, PTNode{PTNode{PTLeaf{VCon{0.0, VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{0.0, VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{0.0, VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 8), VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 12), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 24), VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 28), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}, PTNode{PTLeaf{VCon{vec_get(img, 9), VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{vec_get(img, 13), VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 25), VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{vec_get(img, 29), VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}, PTLeaf{VCon{vec_get(img, 10), VCon{vec_get(img, 11), VCon{0.0, VCon{vec_get(img, 14), VCon{vec_get(img, 15), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{vec_get(img, 26), VCon{vec_get(img, 27), VCon{0.0, VCon{vec_get(img, 30), VCon{vec_get(img, 31), VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}}

# === Forward Inference Computation Graph ===
def forward(+input: Vec) -> Vec:
  +patches_0 : PatchTree = extract_patches_0(input)
  c_0_0 : Vec = conv_channel_patches(patches_0, param_conv_in_weight_f0(), F32.neg(0.1140991))
  c_0_1 : Vec = conv_channel_patches(patches_0, param_conv_in_weight_f1(), F32.neg(0.2630937))
  t0 : Vec = vec_concat(c_0_0, c_0_1)
  +t1 : Vec = vec_relu(t0)
  +patches_2 : PatchTree = extract_patches_2(t1)
  c_2_0 : Vec = conv_channel_patches(patches_2, param_res_block_conv1_weight_f0(), 0.0989451)
  c_2_1 : Vec = conv_channel_patches(patches_2, param_res_block_conv1_weight_f1(), F32.neg(0.03536019))
  t2 : Vec = vec_concat(c_2_0, c_2_1)
  +t3 : Vec = vec_relu(t2)
  +patches_4 : PatchTree = extract_patches_4(t3)
  c_4_0 : Vec = conv_channel_patches(patches_4, param_res_block_conv2_weight_f0(), 0.1984918)
  c_4_1 : Vec = conv_channel_patches(patches_4, param_res_block_conv2_weight_f1(), 0.09181899)
  t4 : Vec = vec_concat(c_4_0, c_4_1)
  t5 : Vec = vec_add(t4, t1)
  +t6 : Vec = vec_relu(t5)
  ch_7_0 : Vec = VCon{vec_get(t6, 0), VCon{vec_get(t6, 1), VCon{vec_get(t6, 2), VCon{vec_get(t6, 3), VCon{vec_get(t6, 4), VCon{vec_get(t6, 5), VCon{vec_get(t6, 6), VCon{vec_get(t6, 7), VCon{vec_get(t6, 8), VCon{vec_get(t6, 9), VCon{vec_get(t6, 10), VCon{vec_get(t6, 11), VCon{vec_get(t6, 12), VCon{vec_get(t6, 13), VCon{vec_get(t6, 14), VCon{vec_get(t6, 15), VNil{}}}}}}}}}}}}}}}}}
  avg_7_0 : F32 = global_avg_pool_channel(ch_7_0, 16)
  ch_7_1 : Vec = VCon{vec_get(t6, 16), VCon{vec_get(t6, 17), VCon{vec_get(t6, 18), VCon{vec_get(t6, 19), VCon{vec_get(t6, 20), VCon{vec_get(t6, 21), VCon{vec_get(t6, 22), VCon{vec_get(t6, 23), VCon{vec_get(t6, 24), VCon{vec_get(t6, 25), VCon{vec_get(t6, 26), VCon{vec_get(t6, 27), VCon{vec_get(t6, 28), VCon{vec_get(t6, 29), VCon{vec_get(t6, 30), VCon{vec_get(t6, 31), VNil{}}}}}}}}}}}}}}}}}
  avg_7_1 : F32 = global_avg_pool_channel(ch_7_1, 16)
  t7 : Vec = VCon{avg_7_0, VCon{avg_7_1, VNil{}}}
  t9 : Vec = linear_layer(t7, param_fc_weight(), param_fc_bias())
  t10 : Vec = vec_softmax(t9)
  t10

# === Main Entrypoint & GPU Parallel Verification ===
def main() -> IO(Unit):
  do IO<Unit>:
    test_x : Vec = VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VNil{}}}}}}}}}}}}}}}}}
    
    # Run forward inference with parallel GPU tree dispatch:
    +out : Vec = forward!(test_x)
    
    predicted_class : U32 = vec_argmax(out)
    
    IO.print("BEND_ONNX_OUTPUT: " ++ vec_show(out))
    IO.print("BEND_PREDICTED_CLASS: " ++ U32.show(predicted_class))
