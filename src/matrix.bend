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

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("Bend Matrix & CNN & Transformer Library Loaded.")
