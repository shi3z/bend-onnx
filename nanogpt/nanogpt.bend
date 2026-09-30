import Base

# ==============================================================================
# Bend Parallel Matrix & Tensor Library (GPU-Accelerated via Interaction Trees)
# Supports: GEMM, Conv2D, ResNet Blocks, LayerNorm, GELU, Softmax, ArgMax
# ==============================================================================

# --- Vector Representation ---
type Vec is Data:
  VNil{}
  VCon{head: F32, tail: Vec}

# Vector length
def vec_len(v: Vec) -> U32:
  match v:
    case VNil{}:
      0
    case VCon{_, t}:
      (1 + vec_len(t) : U32)

# Vector dot product: sum_i (x_i * y_i)
def vec_dot_acc(x: Vec, y: Vec, acc: F32) -> F32:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      vec_dot_acc(xt, yt, (acc + xh * yh : F32))
    case _ _:
      acc

def vec_dot(x: Vec, y: Vec) -> F32:
  vec_dot_acc(x, y, 0.0)

# Vector elementwise addition: x + y (Residual skip connection)
def vec_add(x: Vec, y: Vec) -> Vec:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      VCon{(xh + yh : F32), vec_add(xt, yt)}
    case _ _:
      VNil{}

# Vector elementwise subtraction: x - y
def vec_sub(x: Vec, y: Vec) -> Vec:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      VCon{(xh - yh : F32), vec_sub(xt, yt)}
    case _ _:
      VNil{}

# Vector elementwise multiplication: x * y
def vec_mul(x: Vec, y: Vec) -> Vec:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      VCon{(xh * yh : F32), vec_mul(xt, yt)}
    case _ _:
      VNil{}

# Vector scalar multiplication: s * v
def vec_scale(+s: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(s * h : F32), vec_scale(s, t)}

# Vector concatenation: [x..., y...]
def vec_concat(x: Vec, y: Vec) -> Vec:
  match x:
    case VNil{}:
      y
    case VCon{h, t}:
      VCon{h, vec_concat(t, y)}

# Index into vector with out-of-bounds defaulting to 0.0 (used for zero-padding)
def vec_get(+v: Vec, +idx: U32) -> F32:
  match v idx:
    case VCon{+h, _} 0:
      h
    case VCon{_, t} _:
      vec_get(t, U32.sub(idx, 1))
    case _ _:
      0.0

# --- Activation & Normalization Functions ---

# ReLU: max(0.0, x)
def vec_relu(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      h_relu = F32.max(0.0, h)
      VCon{h_relu, vec_relu(t)}

# Sigmoid: 1.0 / (1.0 + exp(-x))
def f32_sigmoid(x: F32) -> F32:
  neg_x = F32.neg(x)
  e = F32.exp(neg_x)
  denom = (1.0 + e : F32)
  (1.0 / denom : F32)

def vec_sigmoid(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{f32_sigmoid(h), vec_sigmoid(t)}

# GELU Activation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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

# Vector maximum element
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

# Exponential of vector elements shifted by a scalar (for numerical stability)
def vec_exp_shifted(+shift: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      diff = (h - shift : F32)
      VCon{F32.exp(diff), vec_exp_shifted(shift, t)}

# Vector sum
def vec_sum_acc(v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vec_sum_acc(t, (acc + h : F32))

def vec_sum(v: Vec) -> F32:
  vec_sum_acc(v, 0.0)

# Vector division by scalar: v / denom
def vec_scale_div(+denom: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(h / denom : F32), vec_scale_div(denom, t)}

# Numerically stable Softmax: exp(x - max) / sum(exp(x - max))
def vec_softmax(+v: Vec) -> Vec:
  m : F32 = vec_max(v)
  +exps : Vec = vec_exp_shifted(m, v)
  s : F32 = vec_sum(exps)
  vec_scale_div(s, exps)

# Argmax: returns the index (U32) of the largest element
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

# --- Layer Normalization ---
def vec_mean(+v: Vec) -> F32:
  s = vec_sum(v)
  n = U32.to_f32(vec_len(v))
  (s / n : F32)

def vec_sq_diff_acc(+mean: F32, v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      +d = (h - mean : F32)
      vec_sq_diff_acc(mean, t, (acc + d * d : F32))

def vec_var(+v: Vec, +mean: F32) -> F32:
  sq_sum = vec_sq_diff_acc(mean, v, 0.0)
  n = U32.to_f32(vec_len(v))
  (sq_sum / n : F32)

def vec_layernorm_step(+mean: F32, +inv_std: F32, v: Vec, gamma: Vec, beta: Vec) -> Vec:
  match v gamma beta:
    case VCon{+x, xt} VCon{+g, gt} VCon{+b, bt}:
      normed = ((x - mean : F32) * inv_std : F32)
      y = (normed * g + b : F32)
      VCon{y, vec_layernorm_step(mean, inv_std, xt, gt, bt)}
    case _ _ _:
      VNil{}

def vec_layernorm(+v: Vec, gamma: Vec, beta: Vec) -> Vec:
  +m : F32 = vec_mean(v)
  v_var : F32 = vec_var(v, m)
  eps : F32 = 0.00001
  inv_std = (1.0 / F32.sqrt((v_var + eps : F32)) : F32)
  vec_layernorm_step(m, inv_std, v, gamma, beta)

# --- Tree-Structured Matrix Representation ---
type MatTree is Data:
  MTLeaf{vec: Vec}
  MTNode{left: MatTree, right: MatTree}

# GPU-parallel matrix-vector multiplication:
# Computes [dot(row, x) for row in MatTree]
def mat_tree_mul(+x: Vec, m: MatTree) -> Vec:
  match m:
    case MTLeaf{row}:
      VCon{vec_dot(row, x), VNil{}}
    case MTNode{left, right}:
      l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
      vec_concat(l, r)

# Linear Layer (GEMM): Y = W * X + Bias
def linear_layer(+x: Vec, w: MatTree, b: Vec) -> Vec:
  raw : Vec = mat_tree_mul(x, w)
  vec_add(raw, b)

# Linear Layer with ReLU: Y = ReLU(W * X + Bias)
def linear_relu(+x: Vec, w: MatTree, b: Vec) -> Vec:
  out : Vec = linear_layer(x, w, b)
  vec_relu(out)

# Linear Layer with Softmax: Y = Softmax(W * X + Bias)
def linear_softmax(+x: Vec, w: MatTree, b: Vec) -> Vec:
  out : Vec = linear_layer(x, w, b)
  vec_softmax(out)

# --- Convolutional (Conv2D) Trees ---
# Receptive field image patches arranged as a balanced binary tree
type PatchTree is Data:
  PTLeaf{patch: Vec}
  PTNode{left: PatchTree, right: PatchTree}

# Conv2D: Evaluates all spatial patches across GPU lanes concurrently
def conv2d_patches(patches: PatchTree, +w: MatTree, +b: Vec) -> Vec:
  match patches:
    case PTLeaf{patch}:
      out_channels : Vec = mat_tree_mul(patch, w)
      vec_add(out_channels, b)
    case PTNode{left, right}:
      l r = conv2d_patches!(left, w, b) conv2d_patches!(right, w, b)
      vec_concat(l, r)

# Per-channel Conv2D: computes spatial map for one filter in NCHW order
def conv_channel_patches(patches: PatchTree, +filter: Vec, +bias: F32) -> Vec:
  match patches:
    case PTLeaf{patch}:
      VCon{((vec_dot(patch, filter) + bias : F32)), VNil{}}
    case PTNode{left, right}:
      l r = conv_channel_patches!(left, filter, bias) conv_channel_patches!(right, filter, bias)
      vec_concat(l, r)

# Global Average Pooling: computes mean over spatial elements
def global_avg_pool_channel(x: Vec, +count: U32) -> F32:
  (vec_sum(x) / U32.to_f32(count) : F32)

# ResNet Residual Add + ReLU block: ReLU(F(X) + X)
def resnet_add_relu(fx: Vec, x: Vec) -> Vec:
  vec_relu(vec_add(fx, x))

# --- 2D Full Matrix Multiplication (GEMM) ---
def row_cols_mul(+row: Vec, b_cols: MatTree) -> Vec:
  match b_cols:
    case MTLeaf{col}:
      VCon{vec_dot(row, col), VNil{}}
    case MTNode{left, right}:
      l r = row_cols_mul!(row, left) row_cols_mul!(row, right)
      vec_concat(l, r)

type MatResult is Data:
  MRLeaf{row: Vec}
  MRNode{left: MatResult, right: MatResult}

def mat_mul_2d(a_rows: MatTree, +b_cols: MatTree) -> MatResult:
  match a_rows:
    case MTLeaf{row}:
      MRLeaf{row_cols_mul(row, b_cols)}
    case MTNode{left, right}:
      l r = mat_mul_2d!(left, b_cols) mat_mul_2d!(right, b_cols)
      MRNode{l, r}

# --- String Formatting ---
def vec_show_inner(v: Vec) -> String:
  match v:
    case VNil{}:
      ""
    case VCon{+h, t}:
      " " ++ F32.show(h) ++ vec_show_inner(t)

def vec_show(v: Vec) -> String:
  "[" ++ vec_show_inner(v) ++ " ]"

def mat_res_show(m: MatResult) -> String:
  match m:
    case MRLeaf{row}:
      vec_show(row) ++ "\n"
    case MRNode{left, right}:
      mat_res_show(left) ++ mat_res_show(right)

# ==============================================================================
# nanoGPT in Pure Bend (Transformer Architecture with Causal Self-Attention)
# Ported from Andrej Karpathy's nanoGPT
# ==============================================================================

# --- Token List Representation ---
type Tokens is Data:
  TNil{}
  TCon{tok: U32, rest: Tokens}

def tokens_len(ts: Tokens) -> U32:
  match ts:
    case TNil{}:
      0
    case TCon{_, t}:
      (1 + tokens_len(t) : U32)

def tokens_append(ts: Tokens, val: U32) -> Tokens:
  match ts:
    case TNil{}:
      TCon{val, TNil{}}
    case TCon{h, t}:
      TCon{h, tokens_append(t, val)}

def tokens_last(+ts: Tokens) -> U32:
  match ts:
    case TNil{}:
      0
    case TCon{h, TNil{}}:
      h
    case TCon{_, t}:
      tokens_last(t)

# --- Vector List for Embeddings ---
type VecList is Data:
  VLNil{}
  VLCon{head: Vec, tail: VecList}

def vec_list_get(+vs: VecList, +idx: U32) -> Vec:
  match vs idx:
    case VLCon{+row, _} 0:
      row
    case VLCon{_, t} _:
      vec_list_get(t, U32.sub(idx, 1))
    case _ _:
      VNil{}

# --- Key-Value Cache for Causal Attention ---
type KVPair is Data:
  KVP{k: Vec, v: Vec}

type KVList is Data:
  KVNil{}
  KVCon{pair: KVPair, rest: KVList}

# --- Model Parameters (Trained Weights) ---
def param_wte() -> VecList:
  VLCon{VCon{1.504896, VCon{1.289391, VCon{0.5569671, VCon{F32.neg(2.05747), VCon{0.7801255, VCon{F32.neg(1.351272), VCon{F32.neg(0.01158839), VCon{F32.neg(1.462116), VCon{F32.neg(0.8567305), VCon{2.059168, VCon{0.1470846, VCon{F32.neg(1.328366), VCon{F32.neg(0.4272158), VCon{F32.neg(0.8510332), VCon{F32.neg(1.255388), VCon{0.4725104, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.600731, VCon{0.02051508, VCon{F32.neg(0.7493644), VCon{0.4837348, VCon{F32.neg(0.1582135), VCon{1.489555, VCon{1.127244, VCon{2.354927, VCon{0.6837275, VCon{1.036462, VCon{0.8363855, VCon{1.384582, VCon{F32.neg(0.9883273), VCon{F32.neg(0.2737055), VCon{F32.neg(0.7525284), VCon{1.079326, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.14478), VCon{F32.neg(1.026259), VCon{F32.neg(0.7392598), VCon{2.243706, VCon{0.3669661, VCon{0.05196472, VCon{0.5997746, VCon{F32.neg(1.146725), VCon{F32.neg(1.353727), VCon{1.104543, VCon{F32.neg(1.147855), VCon{F32.neg(0.473818), VCon{F32.neg(1.578126), VCon{1.793463, VCon{F32.neg(0.872157), VCon{F32.neg(0.6219052), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.913494), VCon{F32.neg(0.6579012), VCon{0.07799616, VCon{0.5256228, VCon{F32.neg(0.4878165), VCon{1.19094, VCon{F32.neg(0.8137144), VCon{F32.neg(0.7357281), VCon{F32.neg(1.402747), VCon{0.0359907, VCon{F32.neg(0.06345492), VCon{0.6753717, VCon{F32.neg(0.09777202), VCon{1.843936, VCon{F32.neg(1.184108), VCon{1.383049, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.030233, VCon{1.049228, VCon{2.583183, VCon{0.318461, VCon{0.8181444, VCon{F32.neg(0.5348497), VCon{F32.neg(0.8207406), VCon{1.659629, VCon{0.121905, VCon{0.6057135, VCon{F32.neg(0.3623137), VCon{0.0553464, VCon{0.7698993, VCon{F32.neg(0.7731633), VCon{F32.neg(2.123839), VCon{F32.neg(0.8107494), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.3176378, VCon{F32.neg(0.6528871), VCon{F32.neg(1.519512), VCon{F32.neg(0.8357649), VCon{0.6913674, VCon{0.5810749, VCon{1.430629, VCon{0.3433162, VCon{0.563157, VCon{F32.neg(0.408639), VCon{F32.neg(1.408476), VCon{0.7886531, VCon{F32.neg(1.091913), VCon{F32.neg(0.4717864), VCon{1.136694, VCon{0.2671638, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(3.054634), VCon{0.178249, VCon{0.2846483, VCon{F32.neg(0.5337262), VCon{0.04811041, VCon{1.29279, VCon{1.747899, VCon{F32.neg(1.189548), VCon{F32.neg(0.8645409), VCon{1.327944, VCon{1.03816, VCon{0.2149077, VCon{0.5735297, VCon{2.022071, VCon{0.3018484, VCon{0.3458883, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.7481861), VCon{F32.neg(0.2733397), VCon{F32.neg(0.06117864), VCon{F32.neg(0.1730564), VCon{0.6890278, VCon{F32.neg(0.05804042), VCon{F32.neg(0.7020133), VCon{0.2151333, VCon{0.08285584, VCon{0.5690166, VCon{0.9243987, VCon{1.193944, VCon{1.707387, VCon{F32.neg(0.3877504), VCon{0.1169455, VCon{F32.neg(0.03810438), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.930474, VCon{1.011506, VCon{F32.neg(1.435892), VCon{F32.neg(1.129459), VCon{F32.neg(0.1359863), VCon{1.634823, VCon{0.6545047, VCon{0.5757971, VCon{1.141093, VCon{0.01855787, VCon{F32.neg(1.805161), VCon{0.9251059, VCon{F32.neg(0.3752085), VCon{1.032715, VCon{F32.neg(0.6864077), VCon{0.6365848, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.5829306), VCon{0.6122842, VCon{1.93456, VCon{1.172303, VCon{0.2321323, VCon{F32.neg(0.2374936), VCon{F32.neg(0.3291194), VCon{0.2318747, VCon{F32.neg(0.1204245), VCon{0.4258351, VCon{F32.neg(0.6578633), VCon{1.411582, VCon{F32.neg(2.978454), VCon{F32.neg(0.227705), VCon{F32.neg(0.5964078), VCon{0.9553884, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.899869), VCon{0.2284943, VCon{0.02485046, VCon{F32.neg(0.3458251), VCon{0.2867291, VCon{F32.neg(0.7305778), VCon{0.1747577, VCon{F32.neg(1.093543), VCon{F32.neg(1.601588), VCon{1.352411, VCon{1.28837, VCon{0.0522767, VCon{F32.neg(1.546293), VCon{0.7564343, VCon{0.7752405, VCon{2.02582, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.03580465, VCon{0.1205458, VCon{F32.neg(0.8053776), VCon{F32.neg(0.2075017), VCon{F32.neg(0.9316117), VCon{F32.neg(1.590394), VCon{F32.neg(1.135575), VCon{F32.neg(0.5224116), VCon{F32.neg(0.5185874), VCon{F32.neg(1.500733), VCon{F32.neg(1.925968), VCon{0.1278047, VCon{1.022541, VCon{F32.neg(0.5555949), VCon{0.7040224, VCon{0.7096257, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.330473, VCon{F32.neg(1.212916), VCon{0.3278708, VCon{0.3734695, VCon{F32.neg(0.7793139), VCon{0.8178042, VCon{0.5925778, VCon{1.5006, VCon{0.9242433, VCon{1.829815, VCon{F32.neg(1.079065), VCon{F32.neg(1.219899), VCon{F32.neg(0.1305307), VCon{1.750473, VCon{0.986197, VCon{0.5112474, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.8143336), VCon{F32.neg(1.020867), VCon{F32.neg(0.4947483), VCon{F32.neg(0.5920371), VCon{0.1542605, VCon{0.4406097, VCon{F32.neg(0.1482387), VCon{F32.neg(2.317614), VCon{F32.neg(0.3978523), VCon{1.0801, VCon{F32.neg(1.780221), VCon{1.507502, VCon{0.3093177, VCon{F32.neg(0.5001302), VCon{1.034631, VCon{1.689046, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.4079638, VCon{2.047929, VCon{0.4396163, VCon{F32.neg(1.023621), VCon{F32.neg(0.1576176), VCon{0.1738191, VCon{0.1029259, VCon{1.699748, VCon{0.3919129, VCon{F32.neg(1.688732), VCon{F32.neg(2.216164), VCon{F32.neg(1.296394), VCon{F32.neg(0.4167722), VCon{0.0216152, VCon{F32.neg(0.2341911), VCon{0.6697195, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.2256639), VCon{0.2951305, VCon{0.02611207, VCon{0.5293339, VCon{F32.neg(0.7364576), VCon{F32.neg(0.5490862), VCon{F32.neg(0.3471967), VCon{1.461152, VCon{0.7085791, VCon{F32.neg(0.8036541), VCon{0.3420129, VCon{0.4996645, VCon{1.233293, VCon{1.267067, VCon{0.8180426, VCon{F32.neg(0.3308201), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.249141), VCon{0.228725, VCon{F32.neg(1.699947), VCon{F32.neg(1.805349), VCon{F32.neg(2.165866), VCon{0.3549449, VCon{F32.neg(0.3782614), VCon{0.01109854, VCon{F32.neg(1.168772), VCon{F32.neg(0.007568495), VCon{1.520887, VCon{0.2055937, VCon{1.625924, VCon{0.5713469, VCon{F32.neg(1.464586), VCon{0.32559, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.378816), VCon{0.625581, VCon{F32.neg(2.584035), VCon{F32.neg(0.02399106), VCon{F32.neg(0.1221491), VCon{F32.neg(0.7467308), VCon{1.708694, VCon{0.05790213, VCon{1.192551, VCon{1.936607, VCon{0.7284559, VCon{0.9805433, VCon{0.4144456, VCon{1.156148, VCon{0.268958, VCon{F32.neg(0.03661647), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.5872257, VCon{F32.neg(0.7582321), VCon{F32.neg(0.7447735), VCon{F32.neg(1.070717), VCon{F32.neg(0.6086176), VCon{0.505363, VCon{F32.neg(0.6274914), VCon{F32.neg(0.02300915), VCon{1.876704, VCon{0.5957381, VCon{1.65074, VCon{0.1567732, VCon{F32.neg(0.1712311), VCon{0.3388413, VCon{1.313704, VCon{0.5689529, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.2615815, VCon{F32.neg(0.4423643), VCon{0.9456574, VCon{F32.neg(0.9161038), VCon{0.8530942, VCon{0.9848666, VCon{F32.neg(2.972391), VCon{F32.neg(0.9633831), VCon{0.4559967, VCon{F32.neg(1.836327), VCon{F32.neg(0.2857631), VCon{F32.neg(1.22853), VCon{F32.neg(0.2296546), VCon{F32.neg(0.6379907), VCon{0.8529887, VCon{F32.neg(1.230003), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.6300707, VCon{F32.neg(0.5730674), VCon{1.320314, VCon{F32.neg(1.957434), VCon{F32.neg(0.08934277), VCon{0.3337215, VCon{0.1709557, VCon{F32.neg(0.07498433), VCon{F32.neg(0.3403076), VCon{1.097569, VCon{F32.neg(2.320699), VCon{F32.neg(0.5519028), VCon{0.04565461, VCon{0.6597742, VCon{F32.neg(0.06812871), VCon{0.3740127, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.371718, VCon{F32.neg(0.1306372), VCon{0.8216969, VCon{F32.neg(0.1924169), VCon{0.0795838, VCon{F32.neg(1.267954), VCon{0.001619972, VCon{0.01497648, VCon{0.1980922, VCon{1.343192, VCon{F32.neg(0.8062508), VCon{1.346685, VCon{0.5977461, VCon{0.06399869, VCon{0.1911146, VCon{F32.neg(0.5416846), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.7936569), VCon{0.3750961, VCon{0.0878784, VCon{F32.neg(1.24104), VCon{F32.neg(0.320139), VCon{F32.neg(0.8440775), VCon{F32.neg(0.5511463), VCon{1.988246, VCon{1.899633, VCon{1.69448, VCon{0.02807947, VCon{F32.neg(0.1753071), VCon{F32.neg(1.772866), VCon{F32.neg(0.7043911), VCon{F32.neg(0.3945125), VCon{1.886139, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.2183645), VCon{0.1662408, VCon{2.143396, VCon{1.703952, VCon{0.3457761, VCon{0.6422462, VCon{F32.neg(0.2038809), VCon{0.6851242, VCon{F32.neg(0.1396379), VCon{F32.neg(1.180321), VCon{F32.neg(1.282472), VCon{0.4483249, VCon{F32.neg(0.5905229), VCon{0.8537556, VCon{F32.neg(0.4898928), VCon{F32.neg(0.3593285), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.6661378, VCon{F32.neg(0.07423774), VCon{F32.neg(0.2095297), VCon{0.1662623, VCon{1.469774, VCon{F32.neg(0.9387505), VCon{F32.neg(0.6011044), VCon{F32.neg(0.09960447), VCon{F32.neg(0.9848014), VCon{F32.neg(2.487572), VCon{F32.neg(0.331201), VCon{0.8432795, VCon{0.9870969, VCon{F32.neg(0.3318567), VCon{F32.neg(0.8073329), VCon{0.8240709, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.02469098, VCon{F32.neg(1.063762), VCon{F32.neg(0.7599217), VCon{F32.neg(0.4073595), VCon{0.9620213, VCon{F32.neg(0.1425901), VCon{0.1526584, VCon{F32.neg(0.03878852), VCon{0.9442697, VCon{F32.neg(1.581833), VCon{0.9867786, VCon{1.145259, VCon{F32.neg(0.1417614), VCon{F32.neg(0.2762413), VCon{F32.neg(0.193144), VCon{0.776502, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.6836321, VCon{F32.neg(1.324117), VCon{F32.neg(0.5158958), VCon{0.5999697, VCon{F32.neg(0.4700527), VCon{F32.neg(0.608429), VCon{F32.neg(0.04617548), VCon{F32.neg(1.645153), VCon{F32.neg(0.4831558), VCon{F32.neg(0.7400297), VCon{0.3141702, VCon{0.1415052, VCon{1.034446, VCon{F32.neg(0.6262161), VCon{F32.neg(0.5149063), VCon{0.6900396, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.7023664), VCon{1.705408, VCon{F32.neg(0.02508571), VCon{1.091695, VCon{1.318775, VCon{F32.neg(0.4541594), VCon{0.3264932, VCon{1.378553, VCon{1.118386, VCon{F32.neg(0.9730718), VCon{0.05613715, VCon{F32.neg(1.629472), VCon{0.5336285, VCon{0.6473521, VCon{0.2415781, VCon{F32.neg(1.207622), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.383384), VCon{0.4866963, VCon{F32.neg(1.001672), VCon{0.0329372, VCon{F32.neg(0.4290421), VCon{F32.neg(0.9814474), VCon{F32.neg(0.6418297), VCon{0.8262955, VCon{1.590823, VCon{F32.neg(0.1207704), VCon{F32.neg(0.482849), VCon{0.1132577, VCon{0.07712315, VCon{F32.neg(0.9224839), VCon{F32.neg(1.261541), VCon{1.085667, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.096255, VCon{F32.neg(0.6834503), VCon{0.06602012, VCon{F32.neg(0.0007735197), VCon{0.1620032, VCon{1.195529, VCon{F32.neg(1.305681), VCon{F32.neg(1.403471), VCon{F32.neg(1.059348), VCon{0.3056191, VCon{0.4149101, VCon{F32.neg(0.717157), VCon{2.832967, VCon{1.952773, VCon{2.047938, VCon{F32.neg(1.087652), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.621108, VCon{0.8509581, VCon{F32.neg(0.4003266), VCon{F32.neg(0.6086126), VCon{F32.neg(0.5079166), VCon{F32.neg(0.6182685), VCon{F32.neg(1.646454), VCon{F32.neg(1.035838), VCon{F32.neg(0.4501448), VCon{F32.neg(0.07294013), VCon{F32.neg(0.5477618), VCon{F32.neg(1.142138), VCon{F32.neg(0.4485912), VCon{F32.neg(0.03044343), VCon{0.3828953, VCon{F32.neg(0.04475362), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.179527, VCon{F32.neg(0.3313098), VCon{0.6492754, VCon{0.09492454, VCon{F32.neg(0.7523156), VCon{F32.neg(0.6470009), VCon{F32.neg(1.281804), VCon{1.964591, VCon{F32.neg(0.9635052), VCon{F32.neg(2.565879), VCon{0.7093625, VCon{0.8195494, VCon{0.6212242, VCon{0.4230357, VCon{F32.neg(0.3387767), VCon{0.5177874, VNil{}}}}}}}}}}}}}}}}}, VLNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def param_wpe() -> VecList:
  VLCon{VCon{F32.neg(1.12389), VCon{0.03754951, VCon{F32.neg(1.12609), VCon{0.6901299, VCon{1.557829, VCon{0.6885616, VCon{F32.neg(0.4307784), VCon{F32.neg(1.324846), VCon{0.7247956, VCon{F32.neg(0.3544337), VCon{F32.neg(0.08607311), VCon{F32.neg(0.259414), VCon{F32.neg(2.094468), VCon{F32.neg(0.2353019), VCon{F32.neg(1.552515), VCon{F32.neg(0.7761107), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.607508, VCon{F32.neg(1.233546), VCon{0.9796121, VCon{0.598724, VCon{0.5531884, VCon{F32.neg(3.057203), VCon{F32.neg(1.10175), VCon{F32.neg(0.9806077), VCon{F32.neg(0.01348494), VCon{F32.neg(0.1810375), VCon{0.7559048, VCon{F32.neg(0.2043302), VCon{1.616732, VCon{0.7172965, VCon{1.083692, VCon{F32.neg(0.8146187), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.397241), VCon{0.3287582, VCon{F32.neg(1.829177), VCon{0.5628383, VCon{F32.neg(1.166465), VCon{F32.neg(0.2454681), VCon{F32.neg(0.3568184), VCon{1.185889, VCon{2.293227, VCon{F32.neg(1.433839), VCon{1.446811, VCon{1.582741, VCon{F32.neg(2.234721), VCon{F32.neg(0.08004371), VCon{F32.neg(1.305432), VCon{0.4154414, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.444452, VCon{2.814332, VCon{0.7348137, VCon{0.1756444, VCon{0.6705304, VCon{F32.neg(0.5735949), VCon{0.5367709, VCon{F32.neg(0.06783006), VCon{0.7677699, VCon{0.7322134, VCon{0.7469072, VCon{F32.neg(0.3538676), VCon{0.727497, VCon{F32.neg(0.7349879), VCon{F32.neg(0.09243797), VCon{0.5487736, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.5050926), VCon{F32.neg(1.336369), VCon{F32.neg(0.2060679), VCon{F32.neg(0.6074803), VCon{F32.neg(2.222646), VCon{F32.neg(0.2205839), VCon{F32.neg(0.3841715), VCon{2.23291, VCon{F32.neg(0.5574043), VCon{F32.neg(0.6943698), VCon{1.058737, VCon{0.3396899, VCon{F32.neg(0.4272328), VCon{F32.neg(0.4187613), VCon{F32.neg(1.647545), VCon{F32.neg(1.381512), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.4545596, VCon{F32.neg(0.2704738), VCon{1.084934, VCon{F32.neg(0.600969), VCon{F32.neg(0.4077924), VCon{0.4559692, VCon{0.8512213, VCon{F32.neg(1.135049), VCon{0.03722807, VCon{F32.neg(1.299536), VCon{0.4360185, VCon{0.5578634, VCon{0.3185229, VCon{0.2643221, VCon{2.063586, VCon{0.2110791, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.84404), VCon{2.527424, VCon{F32.neg(0.7369787), VCon{1.020178, VCon{0.07955175, VCon{F32.neg(2.742234), VCon{1.539514, VCon{0.5105607, VCon{F32.neg(0.5094035), VCon{1.793739, VCon{1.132249, VCon{0.050712, VCon{0.6343427, VCon{0.4098732, VCon{0.3354033, VCon{0.6845925, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.9565943), VCon{0.5387265, VCon{F32.neg(0.5899432), VCon{F32.neg(0.1625537), VCon{0.8587531, VCon{0.2215569, VCon{0.960237, VCon{F32.neg(1.065601), VCon{F32.neg(1.679854), VCon{F32.neg(0.9982604), VCon{1.245149, VCon{1.104037, VCon{F32.neg(0.04507381), VCon{0.9856849, VCon{F32.neg(0.04125601), VCon{1.298256, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(1.175321), VCon{F32.neg(0.4082016), VCon{1.423123, VCon{F32.neg(0.111997), VCon{0.09203236, VCon{F32.neg(0.4258514), VCon{F32.neg(1.34429), VCon{0.869855, VCon{0.4391312, VCon{0.8521072, VCon{F32.neg(0.6605231), VCon{1.51471, VCon{0.4722207, VCon{0.001267029, VCon{F32.neg(0.9470612), VCon{F32.neg(0.6376649), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.8262698), VCon{F32.neg(0.6490992), VCon{F32.neg(0.5547405), VCon{F32.neg(2.329992), VCon{F32.neg(0.3595088), VCon{1.460706, VCon{0.3474131, VCon{0.04197462, VCon{F32.neg(0.550117), VCon{F32.neg(0.7723153), VCon{F32.neg(0.8771584), VCon{0.4689925, VCon{F32.neg(0.591412), VCon{1.229454, VCon{0.6159599, VCon{F32.neg(1.854209), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.654083, VCon{F32.neg(1.696348), VCon{0.1110681, VCon{F32.neg(0.9848358), VCon{F32.neg(0.5679833), VCon{1.061444, VCon{0.1121712, VCon{F32.neg(0.672285), VCon{F32.neg(3.253316), VCon{F32.neg(0.216231), VCon{0.3576613, VCon{F32.neg(1.628994), VCon{0.4718764, VCon{F32.neg(0.6761865), VCon{0.4308219, VCon{0.920798, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.2238368, VCon{0.5494557, VCon{0.1414577, VCon{2.119222, VCon{0.1428477, VCon{F32.neg(0.253561), VCon{F32.neg(0.3737391), VCon{2.02502, VCon{F32.neg(0.721733), VCon{0.1974974, VCon{1.359359, VCon{F32.neg(0.8543095), VCon{F32.neg(0.8872645), VCon{3.473508, VCon{1.877674, VCon{0.9292772, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.05593675, VCon{0.8651823, VCon{1.907816, VCon{0.9615105, VCon{0.2361194, VCon{0.9661036, VCon{F32.neg(0.7496703), VCon{F32.neg(0.5803919), VCon{F32.neg(0.8958765), VCon{0.2787301, VCon{F32.neg(0.503594), VCon{F32.neg(0.8350859), VCon{0.1657718, VCon{0.3830897, VCon{0.7448879, VCon{0.8687044, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.591431, VCon{0.6448525, VCon{1.251183, VCon{1.828328, VCon{1.23641, VCon{F32.neg(0.09759197), VCon{F32.neg(0.07709964), VCon{0.7218754, VCon{0.5108386, VCon{1.542778, VCon{F32.neg(1.132573), VCon{F32.neg(0.1026414), VCon{0.7313904, VCon{F32.neg(1.453378), VCon{F32.neg(1.675593), VCon{F32.neg(0.269763), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.2251546, VCon{F32.neg(0.07721766), VCon{0.9853441, VCon{1.277879, VCon{0.2880482, VCon{0.868742, VCon{F32.neg(0.8094197), VCon{F32.neg(1.429345), VCon{0.4588515, VCon{0.5307397, VCon{F32.neg(1.360994), VCon{1.955548, VCon{1.76786, VCon{F32.neg(0.9854481), VCon{F32.neg(1.236632), VCon{F32.neg(2.301045), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.001008337), VCon{F32.neg(0.8491253), VCon{F32.neg(1.658806), VCon{0.3061797, VCon{1.181615, VCon{0.3259089, VCon{F32.neg(0.3893075), VCon{2.853351, VCon{0.8240718, VCon{0.798061, VCon{1.88835, VCon{0.5932482, VCon{0.06962941, VCon{F32.neg(1.602862), VCon{F32.neg(0.429668), VCon{0.5759513, VNil{}}}}}}}}}}}}}}}}}, VLNil{}}}}}}}}}}}}}}}}}

def param_ln1_g() -> Vec:
  VCon{0.8319701, VCon{0.8243876, VCon{1.085235, VCon{1.090738, VCon{1.238864, VCon{0.9540256, VCon{1.194335, VCon{1.274386, VCon{1.056888, VCon{1.105811, VCon{1.30545, VCon{1.322517, VCon{0.9085957, VCon{0.8442618, VCon{1.002566, VCon{0.9573449, VNil{}}}}}}}}}}}}}}}}}

def param_ln1_b() -> Vec:
  VCon{F32.neg(0.02378194), VCon{0.01118776, VCon{F32.neg(0.2501449), VCon{F32.neg(0.1818288), VCon{F32.neg(0.08110014), VCon{0.1372675, VCon{F32.neg(0.04060149), VCon{0.04560819, VCon{0.1264642, VCon{0.09139856, VCon{0.1333823, VCon{F32.neg(0.07638995), VCon{0.2734422, VCon{F32.neg(0.06608742), VCon{0.1025495, VCon{F32.neg(0.05447679), VNil{}}}}}}}}}}}}}}}}}

def param_wq() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.07222585), VCon{0.09374823, VCon{F32.neg(0.2428109), VCon{F32.neg(0.1202983), VCon{0.1322336, VCon{F32.neg(0.4162679), VCon{F32.neg(0.264142), VCon{F32.neg(0.2317348), VCon{0.1659743, VCon{0.3802828, VCon{0.3397351, VCon{0.2961046, VCon{0.0308817, VCon{F32.neg(0.2722249), VCon{F32.neg(0.07881889), VCon{F32.neg(0.1740048), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.07402973, VCon{F32.neg(0.1312114), VCon{0.08523286, VCon{0.1415694, VCon{0.1975597, VCon{0.2031868, VCon{F32.neg(0.4643861), VCon{0.2199882, VCon{0.04248519, VCon{F32.neg(0.4442072), VCon{0.09361341, VCon{F32.neg(0.4776528), VCon{0.1196237, VCon{0.05136938, VCon{F32.neg(0.07986832), VCon{0.2249851, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.01548493, VCon{0.2770498, VCon{F32.neg(0.09964425), VCon{0.002979851, VCon{0.357978, VCon{F32.neg(0.09121229), VCon{F32.neg(0.4751226), VCon{0.4283287, VCon{0.5361391, VCon{0.01744273, VCon{F32.neg(0.6254381), VCon{0.133948, VCon{F32.neg(0.4204061), VCon{0.1175837, VCon{F32.neg(0.1458661), VCon{F32.neg(0.3865885), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2296259, VCon{0.7267614, VCon{0.5254477, VCon{0.2408184, VCon{0.5170932, VCon{F32.neg(0.1707578), VCon{F32.neg(0.6273808), VCon{F32.neg(0.6253191), VCon{F32.neg(0.5538068), VCon{0.3128994, VCon{F32.neg(0.04053118), VCon{F32.neg(0.4930051), VCon{0.3854035, VCon{F32.neg(0.2609264), VCon{0.4219148, VCon{0.2611247, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.2324989), VCon{F32.neg(0.3280957), VCon{F32.neg(0.1650049), VCon{0.4317887, VCon{F32.neg(0.5287398), VCon{0.385835, VCon{F32.neg(0.01785912), VCon{0.2865609, VCon{F32.neg(0.02516649), VCon{F32.neg(0.3368344), VCon{F32.neg(0.5487583), VCon{F32.neg(0.3028405), VCon{F32.neg(0.2096999), VCon{0.206871, VCon{0.06306262, VCon{F32.neg(0.3686735), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2790613, VCon{F32.neg(0.2212113), VCon{0.1519645, VCon{0.01209205, VCon{F32.neg(0.708935), VCon{0.4319425, VCon{0.3571706, VCon{0.1803632, VCon{F32.neg(0.1235025), VCon{0.1156879, VCon{0.433881, VCon{0.1008359, VCon{0.2597449, VCon{F32.neg(0.02268508), VCon{0.4965134, VCon{0.03319613, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1416306), VCon{0.2144584, VCon{F32.neg(0.001104634), VCon{F32.neg(0.1995824), VCon{F32.neg(0.1378734), VCon{F32.neg(0.5903437), VCon{F32.neg(0.2800416), VCon{0.4309911, VCon{0.480064, VCon{0.3202896, VCon{0.2822085, VCon{F32.neg(0.2309875), VCon{0.3273547, VCon{0.1235065, VCon{0.3404467, VCon{F32.neg(0.2502087), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1362056, VCon{F32.neg(0.01962168), VCon{0.3735848, VCon{0.422675, VCon{0.1934218, VCon{0.579524, VCon{0.2502976, VCon{F32.neg(0.2938119), VCon{F32.neg(0.507921), VCon{F32.neg(0.1496758), VCon{F32.neg(0.2227973), VCon{F32.neg(0.1668496), VCon{0.006470081, VCon{F32.neg(0.1018903), VCon{0.1219855, VCon{0.0743011, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1231531), VCon{F32.neg(0.2741528), VCon{F32.neg(0.2804757), VCon{0.3552807, VCon{F32.neg(0.1529732), VCon{0.6013567, VCon{0.4818646, VCon{0.418818, VCon{F32.neg(0.07249766), VCon{F32.neg(0.2793191), VCon{F32.neg(0.380508), VCon{F32.neg(0.1027246), VCon{F32.neg(0.04292038), VCon{0.09217849, VCon{F32.neg(0.08894034), VCon{0.3176945, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1143294, VCon{F32.neg(0.04938067), VCon{0.2187477, VCon{0.3628791, VCon{F32.neg(0.4111599), VCon{0.3277344, VCon{0.08085191, VCon{0.139169, VCon{F32.neg(0.4312303), VCon{F32.neg(0.1861955), VCon{F32.neg(0.1930733), VCon{F32.neg(0.2048637), VCon{0.374716, VCon{0.04702051, VCon{0.00433084, VCon{0.3494681, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1635623), VCon{0.3549689, VCon{0.4451803, VCon{0.1063171, VCon{F32.neg(0.0451785), VCon{F32.neg(0.4630827), VCon{F32.neg(0.6368281), VCon{F32.neg(0.1994509), VCon{0.2542135, VCon{F32.neg(0.1353904), VCon{0.1020454, VCon{F32.neg(0.5117666), VCon{0.10719, VCon{F32.neg(0.2924086), VCon{0.07276951, VCon{F32.neg(0.1187363), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2048454), VCon{0.170733, VCon{F32.neg(0.3737735), VCon{F32.neg(0.1188717), VCon{F32.neg(0.5840493), VCon{F32.neg(0.01193204), VCon{0.3450445, VCon{F32.neg(0.04518287), VCon{F32.neg(0.07885516), VCon{0.0275462, VCon{0.4377777, VCon{F32.neg(0.3814987), VCon{F32.neg(0.3534163), VCon{0.04840555, VCon{0.3185535, VCon{0.2439907, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.01976566), VCon{F32.neg(0.0597191), VCon{F32.neg(0.3157836), VCon{F32.neg(0.4551917), VCon{0.3864817, VCon{0.1031533, VCon{0.3167073, VCon{F32.neg(0.2146774), VCon{0.2973073, VCon{0.203468, VCon{0.1189402, VCon{0.5539159, VCon{F32.neg(0.2608158), VCon{F32.neg(0.09870335), VCon{0.04573468, VCon{F32.neg(0.1326533), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.09308983), VCon{0.2066332, VCon{F32.neg(0.000176069), VCon{0.6404995, VCon{0.2405738, VCon{0.3408633, VCon{F32.neg(0.2579865), VCon{0.3459415, VCon{F32.neg(0.2453935), VCon{F32.neg(0.2473414), VCon{F32.neg(0.4053907), VCon{F32.neg(0.2506607), VCon{0.1679354, VCon{F32.neg(0.1324596), VCon{F32.neg(0.208767), VCon{F32.neg(0.1022694), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.0743959), VCon{0.0414163, VCon{0.1831354, VCon{0.1152877, VCon{0.3839945, VCon{0.327327, VCon{F32.neg(0.0905908), VCon{F32.neg(0.3071697), VCon{F32.neg(0.1104499), VCon{F32.neg(0.2109761), VCon{F32.neg(0.4379427), VCon{0.1231367, VCon{F32.neg(0.5582427), VCon{F32.neg(0.3836608), VCon{F32.neg(0.5251367), VCon{0.2074839, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3017779), VCon{F32.neg(0.4367236), VCon{F32.neg(0.4911272), VCon{F32.neg(0.1925341), VCon{F32.neg(0.619283), VCon{F32.neg(0.2153441), VCon{0.5815166, VCon{0.6118993, VCon{0.07305921, VCon{F32.neg(0.005497575), VCon{0.3665656, VCon{0.3641421, VCon{F32.neg(0.4045947), VCon{0.07105293, VCon{F32.neg(0.1096307), VCon{F32.neg(0.1574745), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_wk() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.1984876, VCon{F32.neg(0.1705626), VCon{F32.neg(0.02817015), VCon{F32.neg(0.008289644), VCon{0.22382, VCon{F32.neg(0.2452256), VCon{0.2627921, VCon{0.08050332, VCon{F32.neg(0.08545735), VCon{F32.neg(0.4887478), VCon{F32.neg(0.4839185), VCon{F32.neg(0.1743045), VCon{F32.neg(0.07733394), VCon{F32.neg(0.01230071), VCon{0.1054794, VCon{F32.neg(0.2286634), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1214258), VCon{0.4641863, VCon{0.2468126, VCon{0.1055604, VCon{F32.neg(0.08009379), VCon{0.1429604, VCon{F32.neg(0.1586225), VCon{0.2900932, VCon{0.05869928, VCon{0.112146, VCon{0.2512612, VCon{F32.neg(0.1042331), VCon{F32.neg(0.4236611), VCon{F32.neg(0.415786), VCon{F32.neg(0.2442124), VCon{F32.neg(0.1507169), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2514934, VCon{F32.neg(0.02532616), VCon{F32.neg(0.3308579), VCon{F32.neg(0.3574094), VCon{F32.neg(0.3719046), VCon{0.1504372, VCon{0.4632181, VCon{0.6354496, VCon{F32.neg(0.1183864), VCon{0.1489609, VCon{F32.neg(0.2221947), VCon{0.3593288, VCon{0.04062797, VCon{F32.neg(0.1089429), VCon{F32.neg(0.1187639), VCon{0.6704783, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2669075), VCon{0.3390465, VCon{0.1233246, VCon{0.2037096, VCon{F32.neg(0.2989275), VCon{0.2772472, VCon{F32.neg(0.3401594), VCon{0.1388212, VCon{F32.neg(0.1223776), VCon{F32.neg(0.3769791), VCon{0.09457373, VCon{F32.neg(0.7694583), VCon{F32.neg(0.05105526), VCon{0.4673505, VCon{0.5048001, VCon{F32.neg(0.02221523), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.06136675), VCon{F32.neg(0.1459141), VCon{F32.neg(0.007149553), VCon{0.2279931, VCon{F32.neg(0.08251075), VCon{0.3756453, VCon{0.05670696, VCon{0.1627116, VCon{F32.neg(0.2111021), VCon{0.4665807, VCon{F32.neg(0.20891), VCon{0.03741846, VCon{F32.neg(0.5528153), VCon{0.2345986, VCon{F32.neg(0.5192086), VCon{0.5181069, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1745091), VCon{0.05899256, VCon{0.11209, VCon{0.01905604, VCon{0.3912956, VCon{F32.neg(0.06382315), VCon{F32.neg(0.3030228), VCon{F32.neg(0.4502222), VCon{0.01504132, VCon{0.2679603, VCon{F32.neg(0.6213397), VCon{0.3314334, VCon{0.03251627, VCon{F32.neg(0.03587414), VCon{0.1716606, VCon{0.01357208, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.03013778, VCon{F32.neg(0.3350483), VCon{F32.neg(0.4600985), VCon{F32.neg(0.03730172), VCon{0.3220918, VCon{F32.neg(0.3101797), VCon{0.3932244, VCon{0.5975775, VCon{0.3404402, VCon{F32.neg(0.1206971), VCon{F32.neg(0.6704929), VCon{0.01753948, VCon{F32.neg(0.3961226), VCon{F32.neg(0.10342), VCon{0.1525629, VCon{F32.neg(0.02483577), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3171127), VCon{0.2348098, VCon{F32.neg(0.03743331), VCon{F32.neg(0.06526545), VCon{F32.neg(0.132173), VCon{0.4422774, VCon{F32.neg(0.1738161), VCon{F32.neg(0.4637831), VCon{F32.neg(0.2374382), VCon{F32.neg(0.221051), VCon{0.3266079, VCon{F32.neg(0.5047334), VCon{0.2093047, VCon{0.4059564, VCon{F32.neg(0.02541049), VCon{0.1533522, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.2291927, VCon{0.1105458, VCon{0.1652795, VCon{F32.neg(0.1790582), VCon{0.00395522, VCon{0.1958998, VCon{F32.neg(0.2366952), VCon{0.1550827, VCon{F32.neg(0.4239238), VCon{0.4451022, VCon{0.2940772, VCon{0.3520747, VCon{0.01458009, VCon{0.1111812, VCon{F32.neg(0.3168776), VCon{0.07527659, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.457724), VCon{0.09390166, VCon{0.2662792, VCon{0.1363019, VCon{0.05154091, VCon{0.3065056, VCon{0.05125721, VCon{0.0339895, VCon{F32.neg(0.1934648), VCon{0.02576119, VCon{F32.neg(0.04379068), VCon{F32.neg(0.3392635), VCon{F32.neg(0.01054545), VCon{0.3721421, VCon{F32.neg(0.1247754), VCon{F32.neg(0.5089048), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.1815594, VCon{F32.neg(0.3166372), VCon{F32.neg(0.1443627), VCon{F32.neg(0.02590659), VCon{0.04227807, VCon{0.2268596, VCon{0.0879624, VCon{0.2505729, VCon{0.06958296, VCon{F32.neg(0.5275571), VCon{F32.neg(0.1415199), VCon{F32.neg(0.7991312), VCon{F32.neg(0.185121), VCon{0.06744529, VCon{0.2822862, VCon{F32.neg(0.14341), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.06841754), VCon{F32.neg(0.1126349), VCon{F32.neg(0.08733229), VCon{0.3222499, VCon{0.2066659, VCon{F32.neg(0.5176936), VCon{F32.neg(0.3288991), VCon{F32.neg(0.08322886), VCon{0.4909035, VCon{0.4542805, VCon{0.1165166, VCon{F32.neg(0.2015665), VCon{0.05815045, VCon{0.002216253, VCon{F32.neg(0.0735663), VCon{0.1369359, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.3052588, VCon{F32.neg(0.08932733), VCon{0.1225304, VCon{F32.neg(0.08843314), VCon{0.05319963, VCon{F32.neg(0.1922587), VCon{F32.neg(0.02986381), VCon{F32.neg(0.2930448), VCon{0.3451226, VCon{0.05294987, VCon{0.1929183, VCon{0.5365946, VCon{0.08775613, VCon{F32.neg(0.4213953), VCon{F32.neg(0.007119221), VCon{0.03865624, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.112977, VCon{0.1929874, VCon{F32.neg(0.0480827), VCon{0.08303121, VCon{F32.neg(0.1863482), VCon{0.3302773, VCon{0.2637182, VCon{0.5074423, VCon{F32.neg(0.3145836), VCon{F32.neg(0.1921786), VCon{F32.neg(0.1727151), VCon{F32.neg(0.470614), VCon{F32.neg(0.1523352), VCon{0.3221715, VCon{F32.neg(0.4183364), VCon{0.1181578, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.1121358, VCon{F32.neg(0.1313924), VCon{0.08512799, VCon{F32.neg(0.5193387), VCon{F32.neg(0.4408401), VCon{0.2532776, VCon{0.1397851, VCon{F32.neg(0.1028513), VCon{F32.neg(0.283625), VCon{F32.neg(0.05906317), VCon{0.6365017, VCon{F32.neg(0.4287972), VCon{0.3133416, VCon{0.2828426, VCon{0.1896511, VCon{0.3418025, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.3402437, VCon{0.03532607, VCon{F32.neg(0.1792102), VCon{0.09317724, VCon{0.3225097, VCon{F32.neg(0.54949), VCon{0.1266492, VCon{F32.neg(0.145426), VCon{F32.neg(0.1557456), VCon{0.530039, VCon{F32.neg(0.406222), VCon{0.6941255, VCon{0.3413683, VCon{F32.neg(0.01266939), VCon{F32.neg(0.2481849), VCon{0.07275314, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_wv() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.0713452, VCon{F32.neg(0.2104139), VCon{F32.neg(0.06868435), VCon{F32.neg(0.5397616), VCon{F32.neg(0.2425545), VCon{F32.neg(0.117963), VCon{F32.neg(0.1482345), VCon{F32.neg(0.08868608), VCon{0.225916, VCon{F32.neg(0.02004204), VCon{0.3549336, VCon{0.2719947, VCon{0.3219354, VCon{F32.neg(0.241495), VCon{0.4566037, VCon{0.7096628, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.09794278), VCon{F32.neg(0.2191672), VCon{F32.neg(0.1887926), VCon{0.6802301, VCon{0.4335766, VCon{F32.neg(0.2456322), VCon{F32.neg(0.2266876), VCon{0.01686444, VCon{0.243132, VCon{0.05251901, VCon{0.5020607, VCon{F32.neg(0.1150551), VCon{F32.neg(0.1153024), VCon{F32.neg(0.4353436), VCon{0.1122246, VCon{F32.neg(0.06361336), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.07519617, VCon{F32.neg(0.2393568), VCon{F32.neg(0.01698785), VCon{0.06757003, VCon{F32.neg(0.321078), VCon{0.08306258, VCon{0.1968678, VCon{F32.neg(0.1203394), VCon{0.09032338, VCon{F32.neg(0.3360993), VCon{0.3161825, VCon{F32.neg(0.03723764), VCon{F32.neg(0.3198497), VCon{F32.neg(0.07254066), VCon{0.3073476, VCon{0.3212109, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1192475), VCon{0.03041826, VCon{F32.neg(0.5809808), VCon{0.1470539, VCon{0.06959464, VCon{0.269309, VCon{F32.neg(0.3504566), VCon{0.4955934, VCon{0.3457574, VCon{F32.neg(0.1171537), VCon{0.01963729, VCon{F32.neg(0.08134387), VCon{F32.neg(0.2190379), VCon{F32.neg(0.3066552), VCon{0.163749, VCon{0.1864044, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.294138, VCon{F32.neg(0.36246), VCon{F32.neg(0.2816746), VCon{0.03402366, VCon{F32.neg(0.5800248), VCon{F32.neg(0.1973634), VCon{F32.neg(0.03661737), VCon{0.2166437, VCon{0.3064901, VCon{F32.neg(0.3116575), VCon{F32.neg(0.07686375), VCon{0.1324197, VCon{F32.neg(0.2641703), VCon{F32.neg(0.02034783), VCon{0.3452997, VCon{0.08905029, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3206451), VCon{F32.neg(0.06140728), VCon{F32.neg(0.5309749), VCon{0.3190099, VCon{0.1444712, VCon{F32.neg(0.2429625), VCon{F32.neg(0.1833375), VCon{F32.neg(0.147664), VCon{0.4058489, VCon{0.4031616, VCon{0.269238, VCon{0.4245729, VCon{F32.neg(0.2660126), VCon{F32.neg(0.4495833), VCon{F32.neg(0.06067342), VCon{F32.neg(0.1971736), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.3513876, VCon{F32.neg(0.2018519), VCon{0.4964835, VCon{F32.neg(0.1365004), VCon{F32.neg(0.2177048), VCon{F32.neg(0.1220342), VCon{0.05681521, VCon{F32.neg(0.4419162), VCon{0.2020259, VCon{0.1803101, VCon{F32.neg(0.0655419), VCon{0.3151071, VCon{0.2456716, VCon{F32.neg(0.6042601), VCon{0.08783057, VCon{0.169674, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.01517175), VCon{F32.neg(0.4092137), VCon{F32.neg(0.1670592), VCon{0.1889279, VCon{F32.neg(0.01979068), VCon{0.3939088, VCon{F32.neg(0.0105954), VCon{F32.neg(0.3560691), VCon{F32.neg(0.009850138), VCon{F32.neg(0.01451413), VCon{0.1524375, VCon{0.2799077, VCon{F32.neg(0.1445141), VCon{F32.neg(0.188823), VCon{F32.neg(0.3839006), VCon{F32.neg(0.201762), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.2255485), VCon{F32.neg(0.1822448), VCon{F32.neg(0.107547), VCon{F32.neg(0.1134955), VCon{0.003887437, VCon{F32.neg(0.02622552), VCon{0.08770338, VCon{F32.neg(0.2364933), VCon{F32.neg(0.1867949), VCon{0.269392, VCon{0.4200144, VCon{0.3913697, VCon{0.1633819, VCon{F32.neg(0.1760253), VCon{F32.neg(0.1006953), VCon{F32.neg(0.09871896), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1426275, VCon{F32.neg(0.02377786), VCon{0.2338694, VCon{F32.neg(0.2549575), VCon{F32.neg(0.2008514), VCon{F32.neg(0.01821198), VCon{F32.neg(0.1617034), VCon{0.2565371, VCon{F32.neg(0.13284), VCon{0.4228828, VCon{F32.neg(0.3752266), VCon{F32.neg(0.1447861), VCon{0.117553, VCon{F32.neg(0.0281165), VCon{F32.neg(0.3372203), VCon{0.1262867, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2538418), VCon{0.237622, VCon{0.05318132, VCon{F32.neg(0.6161602), VCon{F32.neg(0.4308308), VCon{F32.neg(0.07717168), VCon{0.5609721, VCon{0.4637233, VCon{F32.neg(0.643689), VCon{F32.neg(0.09000026), VCon{F32.neg(0.2309465), VCon{0.1069327, VCon{0.3029324, VCon{0.1994801, VCon{F32.neg(0.06474453), VCon{0.325942, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.150637, VCon{0.3295179, VCon{0.275675, VCon{F32.neg(0.1196813), VCon{F32.neg(0.1864679), VCon{F32.neg(0.3512637), VCon{F32.neg(0.356791), VCon{0.2750559, VCon{F32.neg(0.1193774), VCon{F32.neg(0.3901916), VCon{F32.neg(0.1409652), VCon{F32.neg(0.01579929), VCon{0.24357, VCon{0.2744854, VCon{0.0890147, VCon{0.215729, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.0663175, VCon{F32.neg(0.04613668), VCon{F32.neg(0.6280231), VCon{0.2218074, VCon{F32.neg(0.1208847), VCon{0.06544992, VCon{F32.neg(0.2783917), VCon{0.3941768, VCon{0.2978722, VCon{0.2498331, VCon{F32.neg(0.2591061), VCon{0.2335086, VCon{F32.neg(0.1962774), VCon{F32.neg(0.2278451), VCon{0.015799, VCon{0.03664562, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.04313507, VCon{0.2374127, VCon{F32.neg(0.05551581), VCon{F32.neg(0.1698766), VCon{0.004188406, VCon{F32.neg(0.1110688), VCon{F32.neg(0.8709266), VCon{F32.neg(0.2917315), VCon{F32.neg(0.1738093), VCon{0.3668903, VCon{0.3767827, VCon{0.1759765, VCon{0.1373519, VCon{F32.neg(0.04350251), VCon{0.1442421, VCon{0.00160875, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2271248, VCon{F32.neg(0.2583595), VCon{0.32945, VCon{F32.neg(0.1952622), VCon{F32.neg(0.2175413), VCon{0.1479036, VCon{F32.neg(0.2535346), VCon{0.2425347, VCon{0.02431305, VCon{0.4337424, VCon{0.1904769, VCon{0.4867783, VCon{0.221975, VCon{F32.neg(0.1723339), VCon{0.06007254, VCon{F32.neg(0.01145946), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2699812, VCon{0.06403723, VCon{0.2221134, VCon{0.1522415, VCon{F32.neg(0.3055789), VCon{0.1487656, VCon{F32.neg(0.01859532), VCon{0.3692737, VCon{F32.neg(0.2547917), VCon{F32.neg(0.4181717), VCon{F32.neg(0.3697824), VCon{0.1838404, VCon{0.2617396, VCon{F32.neg(0.2218678), VCon{0.194261, VCon{F32.neg(0.2037429), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_w_attn_proj() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.3157822, VCon{0.07113399, VCon{F32.neg(0.1722437), VCon{0.08750628, VCon{0.1369386, VCon{0.1930971, VCon{0.1963349, VCon{F32.neg(0.3621922), VCon{0.03085881, VCon{0.1588639, VCon{0.04140193, VCon{0.06262147, VCon{F32.neg(0.2399332), VCon{F32.neg(0.09363984), VCon{0.2396941, VCon{0.5797203, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.07844882), VCon{F32.neg(0.4468341), VCon{0.3057037, VCon{F32.neg(0.03703646), VCon{F32.neg(0.06669211), VCon{F32.neg(0.7238963), VCon{F32.neg(0.1474214), VCon{F32.neg(0.06142086), VCon{F32.neg(0.0702305), VCon{0.0002406995, VCon{0.480053, VCon{F32.neg(0.03820788), VCon{F32.neg(0.1661496), VCon{F32.neg(0.1505214), VCon{F32.neg(0.3763622), VCon{0.1280022, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2246979), VCon{F32.neg(0.2331231), VCon{0.09693089, VCon{F32.neg(0.1706653), VCon{F32.neg(0.3249357), VCon{F32.neg(0.431128), VCon{0.1802886, VCon{F32.neg(0.08513416), VCon{F32.neg(0.2546864), VCon{0.02479863, VCon{0.2631021, VCon{0.2799642, VCon{F32.neg(0.1420503), VCon{0.1610725, VCon{F32.neg(0.4427291), VCon{F32.neg(0.124141), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.07218082, VCon{0.3400247, VCon{F32.neg(0.2472238), VCon{F32.neg(0.05983776), VCon{F32.neg(0.123725), VCon{0.3946267, VCon{F32.neg(0.3060035), VCon{0.2459156, VCon{0.1903728, VCon{F32.neg(0.1728531), VCon{F32.neg(0.1854945), VCon{F32.neg(0.3519712), VCon{F32.neg(0.08235968), VCon{F32.neg(0.05359223), VCon{F32.neg(0.5509271), VCon{F32.neg(0.3140739), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3458746), VCon{0.01688855, VCon{0.01695456, VCon{0.1807028, VCon{F32.neg(0.0135301), VCon{0.131132, VCon{F32.neg(0.1859748), VCon{F32.neg(0.08573019), VCon{0.148563, VCon{0.0007379555, VCon{0.2064804, VCon{0.1264773, VCon{F32.neg(0.09008255), VCon{F32.neg(0.1480434), VCon{F32.neg(0.2487345), VCon{F32.neg(0.03029126), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2085619, VCon{0.0457047, VCon{0.07033485, VCon{0.1382561, VCon{0.05440107, VCon{0.1858291, VCon{0.1369168, VCon{0.1181494, VCon{0.2112361, VCon{F32.neg(0.002965847), VCon{F32.neg(0.07775716), VCon{F32.neg(0.2890231), VCon{F32.neg(0.001162921), VCon{0.1007969, VCon{0.2780485, VCon{F32.neg(0.2631007), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1516619), VCon{F32.neg(0.148586), VCon{F32.neg(0.2210841), VCon{F32.neg(0.4679015), VCon{F32.neg(0.09695666), VCon{0.1836598, VCon{0.4312049, VCon{0.4368176, VCon{0.354167, VCon{F32.neg(0.212261), VCon{F32.neg(0.3591475), VCon{F32.neg(0.2820314), VCon{F32.neg(0.1502864), VCon{F32.neg(0.1010024), VCon{0.006098318, VCon{0.02012456, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.00499826), VCon{F32.neg(0.001205673), VCon{0.2313284, VCon{F32.neg(0.4020554), VCon{0.3871885, VCon{F32.neg(0.1488094), VCon{0.3127912, VCon{F32.neg(0.1309969), VCon{F32.neg(0.1350883), VCon{0.09132791, VCon{0.4519222, VCon{0.2574678, VCon{0.2413673, VCon{F32.neg(0.3573906), VCon{0.5534147, VCon{0.6470674, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3458414), VCon{0.2115317, VCon{F32.neg(0.3913733), VCon{0.4150537, VCon{F32.neg(0.09869806), VCon{F32.neg(0.0326144), VCon{F32.neg(0.2836886), VCon{0.184583, VCon{0.2176717, VCon{0.1775807, VCon{0.1754039, VCon{F32.neg(0.09627882), VCon{0.4908118, VCon{F32.neg(0.4496593), VCon{0.2265314, VCon{F32.neg(0.216481), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3874469), VCon{0.2137016, VCon{F32.neg(0.0455857), VCon{0.5403687, VCon{0.3326683, VCon{0.4265211, VCon{F32.neg(0.2186766), VCon{0.267327, VCon{0.03133518, VCon{0.2905768, VCon{F32.neg(0.3646451), VCon{F32.neg(0.5678848), VCon{0.5106009, VCon{F32.neg(0.3142106), VCon{0.10256, VCon{F32.neg(0.01971677), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.007223067, VCon{0.3702118, VCon{F32.neg(0.000857488), VCon{F32.neg(0.1084761), VCon{F32.neg(0.4515104), VCon{F32.neg(0.04691647), VCon{F32.neg(0.01681229), VCon{F32.neg(0.4106141), VCon{F32.neg(0.1742199), VCon{0.2884264, VCon{0.156512, VCon{0.2260332, VCon{F32.neg(0.01057776), VCon{0.3723139, VCon{F32.neg(0.1797113), VCon{F32.neg(0.1502054), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.02015712), VCon{0.1714372, VCon{0.2158401, VCon{F32.neg(0.1499711), VCon{0.1053578, VCon{0.04739105, VCon{0.4100017, VCon{0.1640584, VCon{0.04229825, VCon{F32.neg(0.180285), VCon{F32.neg(0.5522854), VCon{F32.neg(0.3205334), VCon{F32.neg(0.1797838), VCon{0.1439177, VCon{0.0673238, VCon{0.08345447, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.416476, VCon{F32.neg(0.418295), VCon{0.2826618, VCon{0.210092, VCon{0.3688255, VCon{0.1928885, VCon{0.1593816, VCon{F32.neg(0.2599794), VCon{F32.neg(0.3224026), VCon{F32.neg(0.3431342), VCon{F32.neg(0.06174365), VCon{0.1432792, VCon{0.160915, VCon{0.357983, VCon{0.102271, VCon{0.1851995, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.5735923, VCon{0.06922662, VCon{0.3979114, VCon{0.3269095, VCon{0.2871455, VCon{F32.neg(0.04557753), VCon{0.2797511, VCon{0.281689, VCon{F32.neg(0.1612305), VCon{0.2546223, VCon{0.3347396, VCon{F32.neg(0.2194317), VCon{0.4447983, VCon{F32.neg(0.1633719), VCon{0.6523384, VCon{0.5968475, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2623515), VCon{0.1090203, VCon{F32.neg(0.426741), VCon{F32.neg(0.4867594), VCon{F32.neg(0.6080383), VCon{F32.neg(0.4404769), VCon{F32.neg(0.2107989), VCon{0.2391083, VCon{0.4948148, VCon{0.5899036, VCon{0.2832338, VCon{0.2080894, VCon{F32.neg(0.4262577), VCon{0.4627642, VCon{0.3729805, VCon{F32.neg(0.2447314), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.05208894, VCon{F32.neg(0.3607084), VCon{F32.neg(0.120754), VCon{F32.neg(0.1879374), VCon{0.1789367, VCon{F32.neg(0.2178327), VCon{F32.neg(0.3089253), VCon{F32.neg(0.3119068), VCon{F32.neg(0.3649496), VCon{0.210369, VCon{0.2080624, VCon{0.2517892, VCon{F32.neg(0.3000272), VCon{F32.neg(0.1354803), VCon{0.04812651, VCon{F32.neg(0.2070802), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_ln2_g() -> Vec:
  VCon{1.053342, VCon{0.9292198, VCon{0.838582, VCon{1.255797, VCon{0.8488283, VCon{1.454013, VCon{1.22374, VCon{1.08806, VCon{0.9427604, VCon{1.167268, VCon{1.031826, VCon{1.220308, VCon{1.085253, VCon{1.373629, VCon{1.140491, VCon{1.617101, VNil{}}}}}}}}}}}}}}}}}

def param_ln2_b() -> Vec:
  VCon{F32.neg(0.02890912), VCon{F32.neg(0.1438394), VCon{F32.neg(0.167439), VCon{0.04477389, VCon{0.1023755, VCon{F32.neg(0.003447544), VCon{F32.neg(0.04773455), VCon{0.001396408, VCon{0.08436421, VCon{0.05805185, VCon{F32.neg(0.08957815), VCon{F32.neg(0.09361075), VCon{0.2337839, VCon{F32.neg(0.04046895), VCon{F32.neg(0.2247035), VCon{0.006126129, VNil{}}}}}}}}}}}}}}}}}

def param_w_mlp_fc() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.4709), VCon{0.1897423, VCon{F32.neg(0.2304778), VCon{0.5076585, VCon{F32.neg(0.3287924), VCon{0.1146644, VCon{0.5022534, VCon{F32.neg(0.242502), VCon{0.09623876, VCon{F32.neg(0.3062073), VCon{0.4263742, VCon{0.2469701, VCon{F32.neg(0.0775543), VCon{F32.neg(0.3501078), VCon{F32.neg(0.2768211), VCon{F32.neg(0.2788113), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4110862, VCon{F32.neg(0.3953745), VCon{0.5625509, VCon{F32.neg(0.1886386), VCon{F32.neg(0.6648253), VCon{F32.neg(0.3120063), VCon{F32.neg(0.498253), VCon{F32.neg(0.06759624), VCon{0.2175771, VCon{F32.neg(0.04500627), VCon{F32.neg(0.07729957), VCon{F32.neg(0.1021636), VCon{0.06841975, VCon{0.4095321, VCon{0.2896361, VCon{0.6981076, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.319387, VCon{F32.neg(0.5939612), VCon{F32.neg(0.1518687), VCon{F32.neg(0.007192479), VCon{F32.neg(0.2750487), VCon{0.08979057, VCon{F32.neg(0.03329712), VCon{F32.neg(0.5506757), VCon{0.6403255, VCon{F32.neg(0.7346864), VCon{F32.neg(0.08733153), VCon{F32.neg(0.1116653), VCon{F32.neg(0.07553782), VCon{0.3530309, VCon{0.9927416, VCon{F32.neg(0.8376468), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1857959), VCon{0.3062856, VCon{F32.neg(0.5545911), VCon{F32.neg(0.679469), VCon{0.2511626, VCon{0.64585, VCon{0.2525842, VCon{0.450129, VCon{0.6995196, VCon{F32.neg(0.62148), VCon{F32.neg(0.01519179), VCon{0.0206121, VCon{F32.neg(0.2291023), VCon{0.04794642, VCon{F32.neg(0.03009623), VCon{0.1555775, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3937611), VCon{F32.neg(0.4592558), VCon{F32.neg(0.5321652), VCon{0.7039172, VCon{0.3694806, VCon{F32.neg(0.6117422), VCon{0.7288851, VCon{F32.neg(0.4278627), VCon{0.02028678, VCon{0.6105297, VCon{0.1510505, VCon{0.2817462, VCon{F32.neg(0.02040303), VCon{0.3267448, VCon{0.6400005, VCon{0.1098211, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3511953), VCon{0.1319582, VCon{F32.neg(0.4123783), VCon{0.140972, VCon{0.02498278, VCon{0.5805604, VCon{0.004580302, VCon{F32.neg(0.1066231), VCon{F32.neg(0.0406848), VCon{F32.neg(0.0109133), VCon{0.1906037, VCon{0.2575409, VCon{0.2757984, VCon{F32.neg(0.241863), VCon{F32.neg(0.3458022), VCon{F32.neg(0.006612991), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.3541794, VCon{0.1981903, VCon{0.2678937, VCon{0.6935291, VCon{F32.neg(0.1069747), VCon{F32.neg(0.3466617), VCon{F32.neg(0.3536466), VCon{F32.neg(0.0870311), VCon{0.192983, VCon{0.05126261, VCon{0.006579946, VCon{F32.neg(0.6415397), VCon{0.09422025, VCon{F32.neg(0.5744303), VCon{0.06200851, VCon{F32.neg(0.1625043), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2401172, VCon{0.1813934, VCon{F32.neg(0.03014668), VCon{0.3933263, VCon{F32.neg(0.0312979), VCon{F32.neg(0.322961), VCon{F32.neg(0.3890861), VCon{F32.neg(0.2235528), VCon{0.5291989, VCon{F32.neg(0.4651079), VCon{F32.neg(0.155646), VCon{0.06944207, VCon{0.1373155, VCon{F32.neg(0.04731227), VCon{0.4450492, VCon{F32.neg(0.7933024), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3155129), VCon{F32.neg(0.4477952), VCon{0.1538078, VCon{0.1548418, VCon{F32.neg(0.157009), VCon{F32.neg(0.6406761), VCon{F32.neg(0.2995838), VCon{F32.neg(0.027264), VCon{0.4759651, VCon{0.4614739, VCon{F32.neg(0.2669201), VCon{0.3667839, VCon{F32.neg(0.3486026), VCon{0.5486398, VCon{0.1057958, VCon{F32.neg(0.6200821), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4545664, VCon{F32.neg(0.2700415), VCon{0.2407719, VCon{0.2752258, VCon{0.1786832, VCon{F32.neg(0.3364912), VCon{F32.neg(0.1372566), VCon{0.0730968, VCon{0.234743, VCon{0.06152761, VCon{F32.neg(0.2626291), VCon{F32.neg(0.5521603), VCon{0.2328401, VCon{F32.neg(0.5783216), VCon{0.1866484, VCon{F32.neg(0.1226695), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.615213), VCon{0.01534013, VCon{F32.neg(0.2505678), VCon{0.5664921, VCon{F32.neg(0.1306724), VCon{0.04126333, VCon{0.1654982, VCon{0.380159, VCon{0.4444338, VCon{0.3729724, VCon{F32.neg(0.5155085), VCon{0.02997115, VCon{F32.neg(0.6232522), VCon{0.2748989, VCon{F32.neg(0.557454), VCon{0.09195823, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3724037), VCon{0.200244, VCon{0.1036745, VCon{0.4916943, VCon{0.2048817, VCon{F32.neg(0.03012566), VCon{F32.neg(0.6438586), VCon{0.3602647, VCon{F32.neg(0.1102212), VCon{F32.neg(0.222087), VCon{F32.neg(0.1947159), VCon{F32.neg(0.3536996), VCon{0.2065499, VCon{0.2295119, VCon{0.3776425, VCon{F32.neg(0.2801453), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.3342023, VCon{F32.neg(0.08448298), VCon{0.1003091, VCon{F32.neg(0.3917512), VCon{0.04205994, VCon{0.1420895, VCon{0.2392769, VCon{F32.neg(0.05197236), VCon{F32.neg(0.01049741), VCon{0.1109617, VCon{F32.neg(0.01405904), VCon{F32.neg(0.5535456), VCon{F32.neg(0.07753621), VCon{0.1688754, VCon{F32.neg(0.2081869), VCon{0.2851582, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1107788, VCon{0.3503977, VCon{0.3473151, VCon{0.3082302, VCon{0.09194633, VCon{F32.neg(0.3170093), VCon{F32.neg(0.4384513), VCon{0.05829325, VCon{0.1196415, VCon{F32.neg(0.1630022), VCon{F32.neg(0.08892626), VCon{F32.neg(0.4774762), VCon{0.04067877, VCon{F32.neg(0.6169693), VCon{F32.neg(0.239837), VCon{F32.neg(0.6165302), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2172987), VCon{0.2439852, VCon{0.364197, VCon{0.4828677, VCon{F32.neg(0.3090099), VCon{0.2584453, VCon{F32.neg(0.1612158), VCon{0.3089703, VCon{F32.neg(0.4791003), VCon{F32.neg(0.3497941), VCon{F32.neg(0.2374351), VCon{F32.neg(0.4470846), VCon{F32.neg(0.06590436), VCon{0.1080232, VCon{0.4238012, VCon{0.207429, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1987768, VCon{F32.neg(0.2846405), VCon{0.2767884, VCon{F32.neg(0.3740228), VCon{F32.neg(0.4350737), VCon{F32.neg(0.01989343), VCon{0.2204136, VCon{0.1238158, VCon{0.164384, VCon{0.2753866, VCon{0.1696982, VCon{F32.neg(0.2577912), VCon{0.1870268, VCon{F32.neg(0.4722554), VCon{F32.neg(0.3024038), VCon{0.6988074, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.2308265, VCon{0.05179896, VCon{0.0565389, VCon{0.5084229, VCon{0.2145974, VCon{F32.neg(0.6823626), VCon{F32.neg(0.3044209), VCon{F32.neg(0.362986), VCon{F32.neg(0.2408496), VCon{F32.neg(0.07843586), VCon{F32.neg(0.3827229), VCon{F32.neg(0.2613309), VCon{0.2585314, VCon{0.09834357, VCon{0.2353472, VCon{F32.neg(0.6000227), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2514894), VCon{F32.neg(0.08937553), VCon{F32.neg(0.1321283), VCon{F32.neg(0.2017003), VCon{0.2370891, VCon{0.5613917, VCon{0.5872972, VCon{F32.neg(0.609787), VCon{F32.neg(0.6486607), VCon{0.08933777, VCon{0.2041134, VCon{0.4260697, VCon{0.6261178, VCon{0.2386823, VCon{0.1319502, VCon{0.4125829, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.4656967), VCon{0.4106455, VCon{F32.neg(0.05771538), VCon{0.2991269, VCon{0.3000233, VCon{0.3932406, VCon{F32.neg(0.7123653), VCon{F32.neg(0.5769038), VCon{0.2179752, VCon{F32.neg(0.5910665), VCon{0.2858611, VCon{F32.neg(0.4682012), VCon{0.2122639, VCon{0.6399859, VCon{F32.neg(0.4752806), VCon{F32.neg(0.2347259), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.3779695, VCon{0.4387342, VCon{F32.neg(0.1149334), VCon{F32.neg(0.3175731), VCon{0.1648847, VCon{F32.neg(0.2322759), VCon{0.02638764, VCon{0.1109359, VCon{F32.neg(0.05355449), VCon{0.2919151, VCon{0.3942457, VCon{F32.neg(0.06804807), VCon{F32.neg(0.2570777), VCon{F32.neg(0.5553833), VCon{F32.neg(0.08233757), VCon{0.5159087, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.1931579, VCon{0.1954118, VCon{F32.neg(0.4218841), VCon{F32.neg(0.02261971), VCon{0.2645998, VCon{0.6864144, VCon{0.2199648, VCon{0.6667655, VCon{0.06906731, VCon{F32.neg(0.6096334), VCon{F32.neg(0.07436838), VCon{F32.neg(0.03315325), VCon{F32.neg(0.4219323), VCon{F32.neg(0.4627408), VCon{0.5078738, VCon{F32.neg(0.03843201), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4273813, VCon{F32.neg(0.1964748), VCon{F32.neg(0.2895244), VCon{F32.neg(0.0601484), VCon{F32.neg(0.492187), VCon{F32.neg(0.08861273), VCon{0.2828566, VCon{0.08405995, VCon{0.2889117, VCon{0.2041808, VCon{0.5451515, VCon{0.1674587, VCon{0.4498118, VCon{F32.neg(0.4485111), VCon{0.1740507, VCon{0.8658394, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2565179), VCon{0.2298848, VCon{F32.neg(0.253841), VCon{F32.neg(0.4197641), VCon{F32.neg(0.216712), VCon{0.4458545, VCon{0.1393517, VCon{0.02810048, VCon{0.1484436, VCon{F32.neg(0.1190939), VCon{0.1878724, VCon{0.5172589, VCon{F32.neg(0.01430535), VCon{0.3294159, VCon{F32.neg(0.1244621), VCon{0.1518789, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.09031792), VCon{0.4606872, VCon{0.1904064, VCon{0.4466528, VCon{0.2389206, VCon{F32.neg(0.4198629), VCon{F32.neg(0.612878), VCon{F32.neg(0.1866129), VCon{F32.neg(0.03887457), VCon{F32.neg(0.3030271), VCon{0.1385565, VCon{F32.neg(0.5126132), VCon{F32.neg(0.02345303), VCon{F32.neg(0.1250313), VCon{0.2711125, VCon{F32.neg(0.5713999), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.127247, VCon{0.3370095, VCon{0.100786, VCon{0.007716736, VCon{0.5684471, VCon{0.09635146, VCon{0.4814783, VCon{F32.neg(0.5979701), VCon{F32.neg(0.2852185), VCon{0.03457904, VCon{0.2965843, VCon{0.08057153, VCon{F32.neg(0.9299694), VCon{F32.neg(0.3425705), VCon{F32.neg(0.2763253), VCon{0.5133851, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.212918, VCon{F32.neg(0.006709225), VCon{0.5765681, VCon{F32.neg(0.1233914), VCon{0.01104168, VCon{F32.neg(0.2893496), VCon{F32.neg(0.1783952), VCon{F32.neg(0.02715505), VCon{F32.neg(0.08781571), VCon{0.4956324, VCon{F32.neg(0.5237648), VCon{F32.neg(0.4274618), VCon{0.2169676, VCon{0.1755501, VCon{0.2073895, VCon{0.6370595, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.3866299, VCon{F32.neg(0.2383754), VCon{0.110519, VCon{F32.neg(0.3195819), VCon{F32.neg(0.07319588), VCon{0.364582, VCon{F32.neg(0.02265696), VCon{0.8030676, VCon{F32.neg(0.2116638), VCon{F32.neg(0.1257816), VCon{F32.neg(0.6095273), VCon{0.175707, VCon{0.2567009, VCon{F32.neg(0.1290501), VCon{F32.neg(0.03721216), VCon{F32.neg(0.7600636), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1791841), VCon{0.5420693, VCon{F32.neg(0.1780472), VCon{0.3511595, VCon{0.3137363, VCon{F32.neg(0.7944652), VCon{F32.neg(0.1527215), VCon{F32.neg(0.285077), VCon{0.3668466, VCon{F32.neg(0.6231114), VCon{F32.neg(0.06742771), VCon{F32.neg(0.5185264), VCon{0.6596951, VCon{F32.neg(0.09288162), VCon{0.3277107, VCon{F32.neg(0.486779), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.01663805), VCon{F32.neg(0.09081206), VCon{F32.neg(0.04440196), VCon{0.1955864, VCon{0.07830112, VCon{F32.neg(0.6447819), VCon{F32.neg(0.1948194), VCon{F32.neg(0.2131752), VCon{0.6302281, VCon{F32.neg(0.2854935), VCon{F32.neg(0.1202187), VCon{F32.neg(0.0242893), VCon{0.1217808, VCon{F32.neg(0.5253044), VCon{0.3235031, VCon{F32.neg(0.3954465), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.233385), VCon{F32.neg(0.01809806), VCon{0.0704622, VCon{F32.neg(0.5251305), VCon{F32.neg(0.152834), VCon{0.01712461, VCon{0.1936972, VCon{F32.neg(0.3329108), VCon{0.4462828, VCon{0.1644983, VCon{0.2462079, VCon{F32.neg(0.02966368), VCon{0.2920361, VCon{0.5353963, VCon{F32.neg(0.2009976), VCon{F32.neg(0.2325842), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2020996, VCon{F32.neg(0.08941551), VCon{F32.neg(0.4137667), VCon{0.4199493, VCon{F32.neg(0.1372145), VCon{F32.neg(0.6840737), VCon{0.5053582, VCon{F32.neg(0.3738587), VCon{0.2431651, VCon{F32.neg(0.1996226), VCon{F32.neg(0.03912061), VCon{0.370144, VCon{0.2037315, VCon{F32.neg(0.05893813), VCon{0.3980187, VCon{F32.neg(0.8615061), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.02903868, VCon{F32.neg(0.0654837), VCon{F32.neg(0.1247253), VCon{F32.neg(0.3077119), VCon{F32.neg(0.3296395), VCon{0.3201095, VCon{0.2699658, VCon{0.6527797, VCon{0.5169749, VCon{F32.neg(0.5653707), VCon{F32.neg(0.9032646), VCon{0.07873979, VCon{F32.neg(0.5306528), VCon{F32.neg(0.2894688), VCon{0.3117902, VCon{F32.neg(0.5411732), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }
  }

def param_b_mlp_fc() -> Vec:
  VCon{F32.neg(0.234335), VCon{0.00888471, VCon{F32.neg(0.1846043), VCon{F32.neg(0.3810724), VCon{0.1662857, VCon{F32.neg(0.1721444), VCon{0.1998523, VCon{0.135697, VCon{F32.neg(0.04346303), VCon{0.1237041, VCon{F32.neg(0.2927529), VCon{F32.neg(0.2293451), VCon{F32.neg(0.01552526), VCon{F32.neg(0.161402), VCon{F32.neg(0.2868537), VCon{F32.neg(0.1011612), VCon{0.08129743, VCon{F32.neg(0.3935056), VCon{0.08997402, VCon{0.1345362, VCon{0.01505647, VCon{F32.neg(0.3217603), VCon{0.03315987, VCon{0.1249066, VCon{F32.neg(0.2605175), VCon{F32.neg(0.2614985), VCon{0.04306554, VCon{0.3103402, VCon{F32.neg(0.1700192), VCon{0.07860045, VCon{0.08660863, VCon{F32.neg(0.2919492), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def param_w_mlp_proj() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.2837645, VCon{0.1869619, VCon{0.4003001, VCon{F32.neg(0.3334982), VCon{0.5766696, VCon{F32.neg(0.4339552), VCon{F32.neg(0.4510901), VCon{0.1614682, VCon{0.5787427, VCon{F32.neg(0.3188559), VCon{0.3057309, VCon{0.1943045, VCon{0.02424738, VCon{F32.neg(0.2002056), VCon{F32.neg(0.1762033), VCon{0.1644822, VCon{F32.neg(0.16881), VCon{F32.neg(0.1781314), VCon{F32.neg(0.01635546), VCon{F32.neg(0.1715926), VCon{0.4267979, VCon{F32.neg(0.268255), VCon{F32.neg(0.4003733), VCon{0.21235, VCon{F32.neg(0.115743), VCon{0.13262, VCon{F32.neg(0.313509), VCon{0.276877, VCon{0.0444705, VCon{F32.neg(0.2265136), VCon{F32.neg(0.05086895), VCon{0.3500483, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.3130567, VCon{F32.neg(0.3146921), VCon{F32.neg(0.1268989), VCon{0.4463195, VCon{F32.neg(0.1256833), VCon{0.254467, VCon{0.2519566, VCon{0.1160292, VCon{F32.neg(0.679123), VCon{0.2577188, VCon{0.1023216, VCon{F32.neg(0.103767), VCon{F32.neg(0.2623325), VCon{F32.neg(0.008801545), VCon{F32.neg(0.3294126), VCon{F32.neg(0.1032215), VCon{0.2596606, VCon{0.3283994, VCon{0.5302414, VCon{F32.neg(0.3163958), VCon{F32.neg(0.273732), VCon{0.05379348, VCon{0.1053372, VCon{0.1983636, VCon{F32.neg(0.1767835), VCon{0.1969659, VCon{F32.neg(0.3008892), VCon{F32.neg(0.08046171), VCon{0.5272612, VCon{0.08181655, VCon{F32.neg(0.007173289), VCon{0.1322368, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2769561, VCon{F32.neg(0.4922912), VCon{F32.neg(0.6354511), VCon{0.591436, VCon{F32.neg(0.3338139), VCon{0.4376605, VCon{0.4834622, VCon{0.05092655, VCon{F32.neg(0.6365547), VCon{0.09991729, VCon{F32.neg(0.6510727), VCon{0.7472572, VCon{F32.neg(0.259843), VCon{F32.neg(0.1280597), VCon{0.3497597, VCon{0.02688992, VCon{0.311148, VCon{0.4651123, VCon{0.5232055, VCon{0.1118862, VCon{0.3030439, VCon{0.08225778, VCon{0.3884476, VCon{0.5621459, VCon{0.1256781, VCon{F32.neg(0.0726796), VCon{0.1002238, VCon{0.2352148, VCon{0.2189006, VCon{F32.neg(0.4690968), VCon{0.1362249, VCon{F32.neg(0.06360396), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3592081), VCon{0.4876794, VCon{0.420581, VCon{0.3707408, VCon{F32.neg(0.5260683), VCon{F32.neg(0.2942826), VCon{0.182232, VCon{F32.neg(0.339036), VCon{F32.neg(0.5177328), VCon{0.03033474, VCon{0.2588685, VCon{F32.neg(0.1584493), VCon{0.2249472, VCon{0.01073145, VCon{0.001877531, VCon{0.6385564, VCon{F32.neg(0.2105015), VCon{0.08945698, VCon{F32.neg(0.3351433), VCon{0.4684638, VCon{F32.neg(0.4070618), VCon{F32.neg(0.05239825), VCon{0.054652, VCon{F32.neg(0.325588), VCon{0.4514235, VCon{0.5437058, VCon{0.1202112, VCon{F32.neg(0.4052226), VCon{0.193243, VCon{0.2513874, VCon{F32.neg(0.01430925), VCon{0.081425, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.2721182, VCon{F32.neg(0.1327288), VCon{0.0509479, VCon{F32.neg(0.4914241), VCon{0.0408203, VCon{F32.neg(0.6736643), VCon{0.4345767, VCon{0.07647502, VCon{F32.neg(0.2707926), VCon{0.4498474, VCon{F32.neg(0.5578215), VCon{F32.neg(0.1342827), VCon{F32.neg(0.1112751), VCon{0.2470512, VCon{0.3723569, VCon{F32.neg(0.6011731), VCon{0.3510082, VCon{0.1849715, VCon{0.2879666, VCon{0.5176979, VCon{0.1329417, VCon{F32.neg(0.4670311), VCon{0.02720583, VCon{0.2317961, VCon{0.6879646, VCon{F32.neg(0.1397966), VCon{F32.neg(0.4169864), VCon{0.8843893, VCon{0.4907004, VCon{F32.neg(0.09301618), VCon{0.4036243, VCon{0.4001104, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.06406782), VCon{0.3378491, VCon{F32.neg(0.2860102), VCon{0.01209625, VCon{0.5801206, VCon{F32.neg(0.1398051), VCon{F32.neg(0.4611921), VCon{F32.neg(0.4642625), VCon{0.2728131, VCon{F32.neg(0.4137806), VCon{0.2794666, VCon{F32.neg(0.392282), VCon{0.4170925, VCon{F32.neg(0.3837967), VCon{F32.neg(0.191973), VCon{0.3406059, VCon{F32.neg(0.4439618), VCon{F32.neg(0.03949759), VCon{0.1692413, VCon{F32.neg(0.3693889), VCon{F32.neg(0.1767868), VCon{F32.neg(0.2183039), VCon{0.3554131, VCon{F32.neg(0.3362603), VCon{0.02425939, VCon{F32.neg(0.2328367), VCon{F32.neg(0.4016463), VCon{F32.neg(0.3625951), VCon{F32.neg(0.3735135), VCon{0.5603862, VCon{F32.neg(0.4488751), VCon{F32.neg(0.597701), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.02420725, VCon{F32.neg(0.152597), VCon{F32.neg(0.1597359), VCon{0.04663548, VCon{0.3269994, VCon{0.001554725, VCon{F32.neg(0.03946583), VCon{F32.neg(0.117566), VCon{0.04138158, VCon{0.03146319, VCon{F32.neg(0.6952965), VCon{F32.neg(0.2641977), VCon{F32.neg(0.2320131), VCon{F32.neg(0.1689364), VCon{F32.neg(0.1611722), VCon{F32.neg(0.1938237), VCon{F32.neg(0.04466945), VCon{0.02399872, VCon{0.08355583, VCon{F32.neg(0.01913553), VCon{0.4519095, VCon{0.1984529, VCon{0.1695029, VCon{F32.neg(0.113489), VCon{0.2045429, VCon{F32.neg(0.4651756), VCon{F32.neg(0.2602575), VCon{0.5282177, VCon{F32.neg(0.01767895), VCon{0.1842668, VCon{0.4144776, VCon{F32.neg(0.1613606), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.6997663), VCon{0.4663993, VCon{0.8456979, VCon{F32.neg(0.2933973), VCon{F32.neg(0.5129685), VCon{0.01650806, VCon{0.3654706, VCon{0.575238, VCon{0.1065781, VCon{0.3420925, VCon{F32.neg(0.05765044), VCon{F32.neg(0.2869547), VCon{0.08509158, VCon{0.6366095, VCon{0.3454444, VCon{F32.neg(0.04756837), VCon{0.3191786, VCon{F32.neg(0.9621076), VCon{F32.neg(0.4180313), VCon{0.1814726, VCon{0.5001344, VCon{0.3563128, VCon{F32.neg(0.3465108), VCon{0.1980478, VCon{F32.neg(0.02542924), VCon{F32.neg(0.4441442), VCon{0.2983322, VCon{0.110856, VCon{0.5033031, VCon{F32.neg(0.4025309), VCon{0.3032713, VCon{0.3319023, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.01911181), VCon{F32.neg(0.03652517), VCon{0.2140082, VCon{0.4921588, VCon{F32.neg(0.1384275), VCon{F32.neg(0.1397019), VCon{0.8960474, VCon{F32.neg(0.2711771), VCon{F32.neg(0.1313584), VCon{0.1730533, VCon{0.3466094, VCon{F32.neg(0.07166605), VCon{0.3534153, VCon{0.4820074, VCon{F32.neg(0.2763357), VCon{0.3805723, VCon{F32.neg(0.03864323), VCon{F32.neg(0.22171), VCon{0.2299617, VCon{F32.neg(0.1761754), VCon{F32.neg(0.2685535), VCon{F32.neg(0.1026924), VCon{F32.neg(0.1285535), VCon{F32.neg(0.2000059), VCon{0.3377686, VCon{F32.neg(0.1289006), VCon{0.4587125, VCon{0.2459228, VCon{F32.neg(0.1745588), VCon{F32.neg(0.1067329), VCon{F32.neg(0.2965223), VCon{0.3758639, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.03179252, VCon{F32.neg(0.1316143), VCon{F32.neg(0.6849763), VCon{F32.neg(0.7742374), VCon{0.1026058, VCon{0.3457701, VCon{0.2110156, VCon{F32.neg(0.03897781), VCon{F32.neg(0.2256626), VCon{0.4345343, VCon{F32.neg(0.06940217), VCon{0.5746089, VCon{0.345991, VCon{0.1284902, VCon{0.2948073, VCon{0.4207743, VCon{0.02358908, VCon{0.3279632, VCon{0.6567178, VCon{F32.neg(0.1104155), VCon{F32.neg(0.477466), VCon{0.2689759, VCon{F32.neg(0.2486296), VCon{0.2499598, VCon{F32.neg(0.0420158), VCon{F32.neg(0.2693228), VCon{F32.neg(0.08935126), VCon{0.05763631, VCon{F32.neg(0.1196665), VCon{F32.neg(0.1116233), VCon{F32.neg(0.4599296), VCon{F32.neg(0.4882737), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2048137), VCon{F32.neg(0.05875964), VCon{F32.neg(0.1394419), VCon{F32.neg(0.336662), VCon{F32.neg(0.3976375), VCon{0.3758871, VCon{F32.neg(0.3907656), VCon{F32.neg(0.3572301), VCon{F32.neg(0.06722696), VCon{F32.neg(0.3079086), VCon{0.08296195, VCon{0.2477006, VCon{0.201545, VCon{F32.neg(0.07692456), VCon{F32.neg(0.2386767), VCon{0.3120554, VCon{F32.neg(0.5911027), VCon{0.07280415, VCon{F32.neg(0.1979113), VCon{0.2716225, VCon{F32.neg(0.07147621), VCon{F32.neg(0.2442462), VCon{0.646026, VCon{F32.neg(0.2751827), VCon{0.7019642, VCon{F32.neg(0.1204619), VCon{0.1235842, VCon{F32.neg(0.4650042), VCon{F32.neg(0.2546516), VCon{0.4292049, VCon{F32.neg(0.5928411), VCon{F32.neg(0.5236908), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3583101), VCon{F32.neg(0.1119687), VCon{0.5171797, VCon{0.6245416, VCon{0.03201049, VCon{F32.neg(0.2751784), VCon{0.08248076, VCon{0.3575329, VCon{F32.neg(0.07994122), VCon{F32.neg(0.1325406), VCon{F32.neg(0.6356961), VCon{F32.neg(0.2414988), VCon{F32.neg(0.01941017), VCon{0.06540705, VCon{0.2241376, VCon{F32.neg(0.1436185), VCon{F32.neg(0.03112927), VCon{F32.neg(0.2319714), VCon{F32.neg(0.3328725), VCon{F32.neg(0.09847452), VCon{0.4019954, VCon{F32.neg(0.00299215), VCon{F32.neg(0.510716), VCon{F32.neg(0.4983981), VCon{F32.neg(0.5075354), VCon{0.1630612, VCon{0.1926882, VCon{0.6385372, VCon{0.2636703, VCon{0.07374723, VCon{0.3896048, VCon{0.08547063, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.02365816), VCon{F32.neg(0.3262534), VCon{0.7107658, VCon{F32.neg(0.3055429), VCon{0.4553629, VCon{F32.neg(0.5627161), VCon{F32.neg(0.1018119), VCon{0.5098928, VCon{0.115582, VCon{0.107124, VCon{F32.neg(0.07775355), VCon{0.05489982, VCon{F32.neg(0.306918), VCon{0.3406418, VCon{0.1366589, VCon{F32.neg(0.6306557), VCon{0.6126586, VCon{F32.neg(0.001954541), VCon{F32.neg(0.2622411), VCon{0.3271072, VCon{F32.neg(0.04991613), VCon{F32.neg(0.1894465), VCon{0.04854738, VCon{0.2980617, VCon{0.006209395, VCon{F32.neg(0.3558977), VCon{F32.neg(0.407776), VCon{0.7197708, VCon{0.4412269, VCon{F32.neg(0.03429538), VCon{0.5081645, VCon{F32.neg(0.416114), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4926279, VCon{0.5050457, VCon{F32.neg(0.4228722), VCon{0.06827963, VCon{0.4893594, VCon{F32.neg(0.07152656), VCon{0.1745172, VCon{0.03842248, VCon{0.1526233, VCon{0.3254387, VCon{0.1676081, VCon{F32.neg(0.3383044), VCon{F32.neg(0.04139324), VCon{F32.neg(0.2214849), VCon{F32.neg(0.01614796), VCon{F32.neg(0.0932609), VCon{0.3060406, VCon{0.5732832, VCon{F32.neg(0.6575598), VCon{F32.neg(0.2201428), VCon{F32.neg(0.6825491), VCon{0.06124525, VCon{F32.neg(0.4106889), VCon{F32.neg(0.3854876), VCon{F32.neg(0.6447467), VCon{0.5486363, VCon{F32.neg(0.4377812), VCon{F32.neg(0.807243), VCon{0.102718, VCon{F32.neg(0.1762912), VCon{0.4090567, VCon{0.07244575, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2991731), VCon{0.6073934, VCon{0.1063457, VCon{0.3386428, VCon{F32.neg(0.4399211), VCon{0.2229618, VCon{0.2535125, VCon{F32.neg(0.4700846), VCon{0.6222644, VCon{0.08838826, VCon{0.1617487, VCon{F32.neg(0.2070802), VCon{0.05607596, VCon{0.1288505, VCon{0.5016899, VCon{F32.neg(0.3832483), VCon{0.1725644, VCon{0.4114159, VCon{F32.neg(0.05136869), VCon{F32.neg(0.4020481), VCon{F32.neg(0.5444004), VCon{0.3682593, VCon{F32.neg(0.05006458), VCon{0.03805117, VCon{F32.neg(0.1729646), VCon{0.1797745, VCon{0.295121, VCon{F32.neg(0.5855863), VCon{F32.neg(0.4354259), VCon{0.1755893, VCon{F32.neg(0.1575392), VCon{F32.neg(0.3917032), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.0006381109, VCon{0.1334329, VCon{F32.neg(0.1790783), VCon{0.04396016, VCon{F32.neg(0.2697425), VCon{F32.neg(0.08981746), VCon{F32.neg(0.3600801), VCon{0.2875034, VCon{F32.neg(0.000798311), VCon{F32.neg(0.3949844), VCon{0.7350802, VCon{F32.neg(0.2973713), VCon{0.2438204, VCon{F32.neg(0.732007), VCon{F32.neg(0.4674549), VCon{F32.neg(0.03953856), VCon{F32.neg(0.4655967), VCon{F32.neg(0.09027789), VCon{0.09182552, VCon{0.2060991, VCon{0.5267215, VCon{F32.neg(0.641635), VCon{F32.neg(0.002040189), VCon{F32.neg(0.3871987), VCon{0.5207879, VCon{0.2758054, VCon{0.03215845, VCon{F32.neg(0.4532526), VCon{F32.neg(0.02928357), VCon{F32.neg(0.4686252), VCon{F32.neg(0.09399369), VCon{0.5028024, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_b_mlp_proj() -> Vec:
  VCon{F32.neg(0.1319374), VCon{0.2636053, VCon{0.04906391, VCon{F32.neg(0.1991567), VCon{0.1756226, VCon{F32.neg(0.2680402), VCon{0.1490785, VCon{0.4829798, VCon{F32.neg(0.1263231), VCon{0.07814521, VCon{F32.neg(0.1895746), VCon{F32.neg(0.1221182), VCon{F32.neg(0.1035618), VCon{F32.neg(0.1235643), VCon{F32.neg(0.04402133), VCon{F32.neg(0.2023301), VNil{}}}}}}}}}}}}}}}}}

def param_ln_f_g() -> Vec:
  VCon{1.138708, VCon{1.679674, VCon{1.653492, VCon{1.244314, VCon{1.768353, VCon{1.577337, VCon{1.653738, VCon{1.607633, VCon{1.675285, VCon{1.68268, VCon{1.626197, VCon{1.545221, VCon{1.695864, VCon{1.507513, VCon{1.553511, VCon{1.558259, VNil{}}}}}}}}}}}}}}}}}

def param_ln_f_b() -> Vec:
  VCon{F32.neg(0.1016369), VCon{0.4244741, VCon{0.4364297, VCon{F32.neg(0.6499547), VCon{0.4425286, VCon{F32.neg(0.4944548), VCon{0.6046321, VCon{0.6009088, VCon{F32.neg(0.5267333), VCon{0.5151976, VCon{F32.neg(0.4968207), VCon{0.4743416, VCon{F32.neg(0.2942746), VCon{0.6072503, VCon{F32.neg(0.4919776), VCon{F32.neg(0.07413463), VNil{}}}}}}}}}}}}}}}}}

def param_lm_head() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1297158), VCon{0.6292138, VCon{0.63906, VCon{F32.neg(0.02487895), VCon{0.7046076, VCon{F32.neg(0.7045528), VCon{F32.neg(0.3279102), VCon{0.5345879, VCon{0.4623333, VCon{0.5804012, VCon{F32.neg(0.5465614), VCon{F32.neg(0.5343047), VCon{0.3960553, VCon{F32.neg(0.3838106), VCon{F32.neg(0.4776773), VCon{F32.neg(0.3204879), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.7933587), VCon{0.2485082, VCon{0.3519267, VCon{F32.neg(0.7740243), VCon{F32.neg(0.7061876), VCon{1.05969, VCon{0.5042474, VCon{F32.neg(1.116218), VCon{F32.neg(0.8119761), VCon{0.2796441, VCon{0.6937167, VCon{0.4529865, VCon{0.1191907, VCon{0.6067603, VCon{0.305879, VCon{F32.neg(0.3122484), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.03137799, VCon{F32.neg(0.2709612), VCon{F32.neg(0.2492346), VCon{0.5837526, VCon{F32.neg(0.2847737), VCon{0.4599902, VCon{F32.neg(0.5942035), VCon{F32.neg(0.4203102), VCon{0.407209, VCon{F32.neg(0.5792271), VCon{0.2711113, VCon{F32.neg(0.2464172), VCon{0.647191, VCon{F32.neg(0.3327748), VCon{0.3622918, VCon{0.07925183, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3278732), VCon{F32.neg(0.3435513), VCon{F32.neg(0.07099327), VCon{0.5871462, VCon{F32.neg(0.108281), VCon{0.5721471, VCon{F32.neg(0.4719479), VCon{F32.neg(0.3979398), VCon{0.5873805, VCon{F32.neg(0.2935435), VCon{0.5490854, VCon{F32.neg(0.4533775), VCon{0.4510895, VCon{F32.neg(0.6674902), VCon{0.2645948, VCon{0.0901503, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1393086), VCon{0.7231657, VCon{F32.neg(0.5571257), VCon{0.4648364, VCon{F32.neg(0.6834983), VCon{F32.neg(0.06824533), VCon{F32.neg(0.4658786), VCon{0.2563415, VCon{0.7606153, VCon{F32.neg(0.4005282), VCon{F32.neg(0.5797635), VCon{F32.neg(0.1123409), VCon{F32.neg(0.6632935), VCon{F32.neg(0.1417327), VCon{F32.neg(0.7197922), VCon{0.809078, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.04048922, VCon{F32.neg(0.5314243), VCon{F32.neg(0.4370772), VCon{0.7198927, VCon{0.6347875, VCon{0.7571318, VCon{0.7128652, VCon{F32.neg(0.7723374), VCon{0.2657, VCon{0.739539, VCon{F32.neg(0.7325062), VCon{F32.neg(0.5721729), VCon{F32.neg(0.6399413), VCon{0.0839073, VCon{F32.neg(0.2061445), VCon{F32.neg(0.59974), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1228062), VCon{F32.neg(0.1627709), VCon{0.5138639, VCon{0.08095779, VCon{0.9112786, VCon{F32.neg(0.2142459), VCon{0.2662757, VCon{F32.neg(1.011963), VCon{F32.neg(0.7046812), VCon{0.1597537, VCon{0.9358217, VCon{F32.neg(0.4511642), VCon{0.286687, VCon{F32.neg(0.695949), VCon{0.1550798, VCon{0.5573066, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.3815491, VCon{0.8063735, VCon{0.0402933, VCon{F32.neg(0.3941229), VCon{F32.neg(0.1024617), VCon{0.2149093, VCon{F32.neg(0.3024884), VCon{F32.neg(0.4636276), VCon{F32.neg(0.9951149), VCon{F32.neg(0.5717727), VCon{0.826933, VCon{F32.neg(0.2440235), VCon{0.7128848, VCon{0.3441081, VCon{F32.neg(0.4033749), VCon{0.8587551, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1694152), VCon{F32.neg(0.4493944), VCon{F32.neg(0.3006643), VCon{0.3269405, VCon{F32.neg(0.1457192), VCon{0.5922191, VCon{F32.neg(0.7025878), VCon{F32.neg(0.4844813), VCon{0.5848641, VCon{F32.neg(0.3376166), VCon{0.5329587, VCon{F32.neg(0.2926379), VCon{0.5917612, VCon{F32.neg(0.610364), VCon{0.3331573, VCon{F32.neg(0.0003196454), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4769787, VCon{F32.neg(0.9615994), VCon{F32.neg(0.07835308), VCon{F32.neg(0.1422835), VCon{F32.neg(0.4596288), VCon{0.5461318, VCon{0.7071627, VCon{0.3388341, VCon{F32.neg(0.5100982), VCon{0.7427365, VCon{0.7249036, VCon{F32.neg(0.2578571), VCon{F32.neg(0.7098934), VCon{F32.neg(0.949522), VCon{F32.neg(0.2889816), VCon{0.2858958, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1666836), VCon{F32.neg(0.406704), VCon{F32.neg(0.3665811), VCon{0.5937377, VCon{F32.neg(0.1927321), VCon{0.3179272, VCon{F32.neg(0.3341644), VCon{F32.neg(0.4614626), VCon{0.6468921, VCon{F32.neg(0.4843815), VCon{0.6526623, VCon{F32.neg(0.4747057), VCon{0.4530246, VCon{F32.neg(0.4789255), VCon{0.3585174, VCon{F32.neg(0.2020022), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.03483636), VCon{F32.neg(0.3988613), VCon{0.004152458, VCon{0.6758564, VCon{0.06788602, VCon{0.6291248, VCon{F32.neg(0.347592), VCon{F32.neg(0.4479246), VCon{0.6118714, VCon{F32.neg(0.1842767), VCon{0.5420957, VCon{F32.neg(0.4000529), VCon{0.7776429, VCon{F32.neg(0.5016835), VCon{F32.neg(0.0513193), VCon{0.02165942, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.7188025, VCon{F32.neg(0.8319358), VCon{F32.neg(0.9650063), VCon{F32.neg(0.0896833), VCon{F32.neg(0.4718209), VCon{0.8783373, VCon{F32.neg(0.1226987), VCon{0.6834165, VCon{0.6004453, VCon{0.4216397, VCon{F32.neg(0.4710269), VCon{0.1526238, VCon{F32.neg(0.472799), VCon{0.8016442, VCon{0.5548089, VCon{F32.neg(0.001129347), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2153037), VCon{F32.neg(0.3718546), VCon{F32.neg(0.4521905), VCon{0.3399295, VCon{F32.neg(0.1981752), VCon{0.5297607, VCon{F32.neg(0.591479), VCon{F32.neg(0.6363013), VCon{0.4779342, VCon{F32.neg(0.4667305), VCon{0.3976228, VCon{F32.neg(0.2836257), VCon{0.4239103, VCon{F32.neg(0.5731268), VCon{0.1502438, VCon{F32.neg(0.03123004), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.4890847, VCon{F32.neg(0.3013441), VCon{F32.neg(0.1966543), VCon{F32.neg(0.3872419), VCon{0.4374367, VCon{F32.neg(0.4189128), VCon{0.7376652, VCon{0.102769, VCon{F32.neg(0.4109328), VCon{0.1685997, VCon{F32.neg(0.7711329), VCon{0.7969308, VCon{0.80169, VCon{0.59676, VCon{F32.neg(0.1104811), VCon{F32.neg(0.6471182), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4783792, VCon{F32.neg(0.2949283), VCon{0.4662287, VCon{F32.neg(0.03475299), VCon{F32.neg(0.7360864), VCon{0.2881638, VCon{F32.neg(0.02447329), VCon{0.4730453, VCon{F32.neg(0.4441894), VCon{0.1921489, VCon{0.530454, VCon{0.005240434, VCon{F32.neg(0.0860856), VCon{F32.neg(0.3925167), VCon{F32.neg(0.7073112), VCon{F32.neg(0.6797728), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.6119177), VCon{F32.neg(0.4928701), VCon{0.5286135, VCon{F32.neg(0.5118803), VCon{F32.neg(0.560713), VCon{0.169568, VCon{F32.neg(0.5408947), VCon{0.5895417, VCon{F32.neg(0.1580835), VCon{0.547321, VCon{0.5105913, VCon{0.8701897, VCon{0.5663458, VCon{F32.neg(0.446592), VCon{F32.neg(0.6275031), VCon{F32.neg(0.5075109), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1330732), VCon{F32.neg(0.4792791), VCon{F32.neg(0.3131976), VCon{0.5983869, VCon{F32.neg(0.293342), VCon{0.2354065, VCon{F32.neg(0.6340627), VCon{F32.neg(0.6487268), VCon{0.5529743, VCon{F32.neg(0.4819091), VCon{0.6107482, VCon{F32.neg(0.2441402), VCon{0.507138, VCon{F32.neg(0.5840609), VCon{0.3077638, VCon{F32.neg(0.1181401), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2013305, VCon{0.1933301, VCon{F32.neg(0.8087979), VCon{0.1514148, VCon{F32.neg(0.7759985), VCon{0.1794258, VCon{0.579707, VCon{0.3471953, VCon{F32.neg(0.5268022), VCon{0.7354501, VCon{0.3782524, VCon{0.3185855, VCon{0.2357083, VCon{0.01331184, VCon{F32.neg(0.9402451), VCon{0.2249071, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1258306), VCon{F32.neg(0.07796554), VCon{0.1639194, VCon{F32.neg(0.3927868), VCon{F32.neg(0.03040762), VCon{0.2235329, VCon{0.6280094, VCon{0.6486288, VCon{F32.neg(0.3045724), VCon{F32.neg(0.9863265), VCon{F32.neg(0.2490713), VCon{0.6037373, VCon{F32.neg(0.8432074), VCon{F32.neg(0.243598), VCon{0.6531744, VCon{0.6633568, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.8424793, VCon{F32.neg(0.4710344), VCon{0.7521643, VCon{F32.neg(0.4703889), VCon{0.5213825, VCon{0.3859046, VCon{F32.neg(0.9572086), VCon{F32.neg(0.04159561), VCon{F32.neg(0.8021238), VCon{F32.neg(0.4135427), VCon{0.152966, VCon{F32.neg(0.4791864), VCon{0.7746795, VCon{F32.neg(0.8069212), VCon{0.9548805, VCon{F32.neg(0.422594), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3926818), VCon{F32.neg(0.2485017), VCon{F32.neg(0.3772737), VCon{F32.neg(0.2581794), VCon{F32.neg(0.9779153), VCon{0.847823, VCon{0.4705798, VCon{0.1368376, VCon{F32.neg(0.1128622), VCon{F32.neg(0.9160228), VCon{0.6809989, VCon{0.9658362, VCon{0.3875403, VCon{0.1912505, VCon{1.147917, VCon{F32.neg(0.6662923), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.06545408), VCon{F32.neg(0.1478325), VCon{F32.neg(0.23211), VCon{0.5414281, VCon{F32.neg(0.201097), VCon{0.5128319, VCon{F32.neg(0.4957244), VCon{F32.neg(0.7074599), VCon{0.4885909, VCon{F32.neg(0.525145), VCon{0.2560974, VCon{F32.neg(0.2849745), VCon{0.4275707, VCon{F32.neg(0.4483089), VCon{0.2705094, VCon{F32.neg(0.1502746), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.04944801, VCon{F32.neg(0.202247), VCon{F32.neg(0.4646988), VCon{0.4816317, VCon{F32.neg(0.2182629), VCon{0.58341, VCon{F32.neg(0.6259314), VCon{F32.neg(0.5432453), VCon{0.3803341, VCon{F32.neg(0.570949), VCon{0.3964141, VCon{F32.neg(0.2066429), VCon{0.708648, VCon{F32.neg(0.4599472), VCon{0.2100381, VCon{0.2977721, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.05454008), VCon{F32.neg(0.373886), VCon{F32.neg(0.4007133), VCon{0.4304149, VCon{F32.neg(0.2318704), VCon{0.5115443, VCon{F32.neg(0.5418845), VCon{F32.neg(0.3533386), VCon{0.6246849, VCon{F32.neg(0.3975397), VCon{0.3707721, VCon{F32.neg(0.4065706), VCon{0.5874001, VCon{F32.neg(0.6243301), VCon{0.1817891, VCon{F32.neg(0.0198372), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2135502), VCon{F32.neg(0.4551612), VCon{F32.neg(0.2817998), VCon{0.2711706, VCon{F32.neg(0.3041137), VCon{0.5708082, VCon{F32.neg(0.6603714), VCon{F32.neg(0.485695), VCon{0.3950781, VCon{F32.neg(0.445322), VCon{0.2438075, VCon{F32.neg(0.332535), VCon{0.3369489, VCon{F32.neg(0.4412528), VCon{0.1081181, VCon{0.06581069, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.004448262), VCon{F32.neg(0.3896542), VCon{F32.neg(0.2910072), VCon{0.4391531, VCon{F32.neg(0.284734), VCon{0.549596, VCon{F32.neg(0.4098947), VCon{F32.neg(0.3703825), VCon{0.5591415, VCon{F32.neg(0.2866774), VCon{0.4069201, VCon{F32.neg(0.4639272), VCon{0.6441072, VCon{F32.neg(0.4231044), VCon{0.3230833, VCon{0.1708493, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2433308, VCon{F32.neg(0.4184905), VCon{0.4549672, VCon{0.3820413, VCon{F32.neg(0.1900966), VCon{F32.neg(0.6722713), VCon{F32.neg(0.6252785), VCon{0.3552811, VCon{F32.neg(0.5995997), VCon{0.5954944, VCon{F32.neg(0.6489933), VCon{F32.neg(0.5908732), VCon{F32.neg(0.6354493), VCon{0.7081618, VCon{0.9243085, VCon{0.1286061, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1503552), VCon{F32.neg(0.3445009), VCon{F32.neg(0.4528233), VCon{0.4286352, VCon{F32.neg(0.2775527), VCon{0.4585956, VCon{F32.neg(0.4723517), VCon{F32.neg(0.6480164), VCon{0.5634006, VCon{F32.neg(0.4528938), VCon{0.2335317, VCon{F32.neg(0.3817472), VCon{0.4930515, VCon{F32.neg(0.6333139), VCon{0.2859895, VCon{0.0850351, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.02910253), VCon{F32.neg(0.4187999), VCon{F32.neg(0.2759257), VCon{0.3638532, VCon{F32.neg(0.3037852), VCon{0.35162, VCon{F32.neg(0.5177523), VCon{F32.neg(0.4597195), VCon{0.602249, VCon{F32.neg(0.4930893), VCon{0.360494, VCon{F32.neg(0.3629023), VCon{0.5073334, VCon{F32.neg(0.4797523), VCon{0.141879, VCon{0.09943859, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.06096171, VCon{F32.neg(0.3438619), VCon{F32.neg(0.385872), VCon{0.3917746, VCon{F32.neg(0.325549), VCon{0.3496649, VCon{F32.neg(0.3053012), VCon{F32.neg(0.4740295), VCon{0.425613, VCon{F32.neg(0.3289068), VCon{0.3169661, VCon{F32.neg(0.4414379), VCon{0.5168504, VCon{F32.neg(0.6186213), VCon{0.4196628, VCon{F32.neg(0.01126839), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2609713), VCon{F32.neg(0.2907408), VCon{F32.neg(0.1339539), VCon{0.5391462, VCon{F32.neg(0.2395316), VCon{0.5046329, VCon{F32.neg(0.4310465), VCon{F32.neg(0.6877792), VCon{0.8305447, VCon{F32.neg(0.302954), VCon{0.5512391, VCon{F32.neg(0.459531), VCon{0.4362487, VCon{F32.neg(0.6603041), VCon{0.1231874, VCon{F32.neg(0.0532076), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }
  }

# Embedding lookup: WTE[tok] + WPE[pos]
def get_embedding(+tok: U32, +pos: U32) -> Vec:
  tok_vec = vec_list_get(param_wte(), tok)
  pos_vec = vec_list_get(param_wpe(), pos)
  vec_add(tok_vec, pos_vec)

# Compute Key and Value vectors for all past tokens
def compute_kv_list(ts: Tokens, +pos: U32) -> KVList:
  match ts:
    case TNil{}:
      KVNil{}
    case TCon{tok, rest}:
      +x = get_embedding(tok, pos)
      +normed = vec_layernorm(x, param_ln1_g(), param_ln1_b())
      k = mat_tree_mul(normed, param_wk())
      v = mat_tree_mul(normed, param_wv())
      KVCon{KVP{k, v}, compute_kv_list(rest, (pos + 1 : U32))}

# Compute attention score for each past token against query Q
def compute_scores(+q: Vec, kvs: KVList) -> Vec:
  match kvs:
    case KVNil{}:
      VNil{}
    case KVCon{KVP{+k, _}, rest}:
      scale : F32 = 0.25 # 1.0 / sqrt(16.0)
      score = ((vec_dot(q, k) * scale : F32))
      VCon{score, compute_scores(q, rest)}

# Aggregate values weighted by softmax attention weights
def apply_attention(weights: Vec, kvs: KVList) -> Vec:
  match weights kvs:
    case VCon{+w, wt} KVCon{KVP{_, +v}, kt}:
      scaled_v = vec_scale(w, v)
      rest_v = apply_attention(wt, kt)
      vec_add(scaled_v, rest_v)
    case _ _:
      VNil{}

# Transformer forward pass for next-token prediction
def gpt_forward_step(+tokens: Tokens) -> Vec:
  # 1. Fetch current token and position embedding
  curr_tok : U32 = tokens_last(tokens)
  pos : U32 = U32.sub(tokens_len(tokens), 1)
  +x : Vec = get_embedding(curr_tok, pos)
  
  # 2. LayerNorm 1
  normed1 : Vec = vec_layernorm(x, param_ln1_g(), param_ln1_b())
  
  # 3. Compute Query at current position
  +q : Vec = mat_tree_mul(normed1, param_wq())
  
  # 4. Multi-token Causal Self-Attention over past tokens
  +kvs : KVList = compute_kv_list(tokens, 0)
  scores : Vec = compute_scores(q, kvs)
  +attn_weights : Vec = vec_softmax(scores)
  context : Vec = apply_attention(attn_weights, kvs)
  attn_out : Vec = mat_tree_mul(context, param_w_attn_proj())
  
  # Residual connection 1: x + attn_out
  +x1 : Vec = vec_add(x, attn_out)
  
  # 5. LayerNorm 2
  normed2 : Vec = vec_layernorm(x1, param_ln2_g(), param_ln2_b())
  
  # 6. MLP Block: GELU(W_fc * x + b) -> W_proj * h + b
  h_fc : Vec = linear_layer(normed2, param_w_mlp_fc(), param_b_mlp_fc())
  h_gelu : Vec = vec_gelu(h_fc)
  mlp_out : Vec = linear_layer(h_gelu, param_w_mlp_proj(), param_b_mlp_proj())
  
  # Residual connection 2: x1 + mlp_out
  +x2 : Vec = vec_add(x1, mlp_out)
  
  # 7. Final LayerNorm & LM Head (Projection to Vocab Logits)
  normed_final : Vec = vec_layernorm(x2, param_ln_f_g(), param_ln_f_b())
  mat_tree_mul(normed_final, param_lm_head())

# Decode single token ID to character
def token_to_char(tok: U32) -> String:
  match tok:
    case 0:  " "
    case 1:  "A"
    case 2:  "B"
    case 3:  "C"
    case 4:  "D"
    case 5:  "E"
    case 6:  "F"
    case 7:  "G"
    case 8:  "H"
    case 9:  "I"
    case 10: "J"
    case 11: "K"
    case 12: "L"
    case 13: "M"
    case 14: "N"
    case 15: "O"
    case 16: "P"
    case 17: "Q"
    case 18: "R"
    case 19: "S"
    case 20: "T"
    case 21: "U"
    case 22: "V"
    case 23: "W"
    case 24: "X"
    case 25: "Y"
    case 26: "Z"
    case 27: "!"
    case 28: "."
    case 29: ":"
    case 30: "0"
    case _:  "1"

def print_tokens(ts: Tokens) -> String:
  match ts:
    case TNil{}:
      ""
    case TCon{tok, rest}:
      token_to_char(tok) ++ print_tokens(rest)

# Autoregressive generation loop in Bend:
def generate_tokens(steps: Nat, +ts: Tokens) -> Tokens:
  match steps:
    case 0n:
      ts
    case 1n+p:
      +logits : Vec = gpt_forward_step(ts)
      next_tok : U32 = vec_argmax(logits)
      new_tokens : Tokens = tokens_append(ts, next_tok)
      generate_tokens(p, new_tokens)

def main() -> IO(Unit):
  do IO<Unit>:
    # Prompt: "BEND " -> Token IDs: [2, 5, 14, 4, 0]
    +prompt : Tokens = TCon{2, TCon{5, TCon{14, TCon{4, TCon{0, TNil{}}}}}}
    
    IO.print("--- nanoGPT Text Generation in Bend ---")
    IO.print("Prompt:   " ++ print_tokens(prompt))
    
    # Autoregressively generate next 9 tokens using GPU tree operations:
    gen : Tokens = generate_tokens(9n, prompt)
    
    IO.print("Output:   " ++ print_tokens(gen))
    IO.print("---------------------------------------")
