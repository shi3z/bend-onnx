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
# Ported directly from Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)
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

# --- Model Parameters (Karpathy nanoGPT Weights) ---
def param_wte() -> VecList:
  VLCon{VCon{0.230509, VCon{F32.neg(0.569929), VCon{0.631178, VCon{F32.neg(0.538692), VCon{F32.neg(0.194463), VCon{0.102079, VCon{0.618830, VCon{0.579975, VCon{F32.neg(0.587564), VCon{0.483350, VCon{0.421099, VCon{0.341520, VCon{F32.neg(0.551534), VCon{F32.neg(0.232892), VCon{F32.neg(0.494657), VCon{0.286770, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.729684), VCon{0.362953, VCon{0.118031, VCon{0.328889, VCon{F32.neg(0.369403), VCon{F32.neg(0.609305), VCon{F32.neg(0.558186), VCon{0.015950, VCon{0.577762, VCon{0.866090, VCon{0.243268, VCon{0.256674, VCon{F32.neg(0.781009), VCon{F32.neg(0.036647), VCon{0.594819, VCon{F32.neg(0.596416), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.518690, VCon{0.158286, VCon{0.047624, VCon{0.344778, VCon{0.154769, VCon{0.079137, VCon{F32.neg(0.126148), VCon{F32.neg(0.239850), VCon{0.099910, VCon{F32.neg(0.258388), VCon{F32.neg(0.473019), VCon{F32.neg(0.386008), VCon{0.140121, VCon{F32.neg(0.074918), VCon{0.359858, VCon{F32.neg(0.358449), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.501833, VCon{F32.neg(0.060193), VCon{F32.neg(0.263806), VCon{0.379527, VCon{0.023275, VCon{0.306733, VCon{F32.neg(0.431119), VCon{F32.neg(0.256226), VCon{0.119552, VCon{F32.neg(0.504685), VCon{F32.neg(0.428595), VCon{F32.neg(0.490704), VCon{0.489143, VCon{0.453012, VCon{0.420321, VCon{F32.neg(0.162308), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.742825), VCon{F32.neg(0.518810), VCon{F32.neg(0.626362), VCon{F32.neg(0.429438), VCon{0.765808, VCon{F32.neg(0.429393), VCon{F32.neg(0.156085), VCon{0.403783, VCon{F32.neg(0.296399), VCon{F32.neg(0.142205), VCon{0.187261, VCon{F32.neg(0.148112), VCon{F32.neg(0.281072), VCon{0.641167, VCon{F32.neg(0.240958), VCon{0.617954, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.567932, VCon{0.643967, VCon{F32.neg(0.411150), VCon{0.409205, VCon{0.570523, VCon{F32.neg(0.128715), VCon{0.471955, VCon{F32.neg(0.614011), VCon{0.527731, VCon{0.239032, VCon{F32.neg(0.463628), VCon{F32.neg(0.420642), VCon{0.427739, VCon{F32.neg(0.639298), VCon{F32.neg(0.593195), VCon{0.181368, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{1.073671, VCon{0.013477, VCon{0.698626, VCon{0.656006, VCon{F32.neg(0.721525), VCon{F32.neg(0.753895), VCon{F32.neg(0.199103), VCon{0.302323, VCon{F32.neg(0.287389), VCon{F32.neg(0.176547), VCon{0.047733, VCon{0.271530, VCon{F32.neg(0.306153), VCon{0.205975, VCon{0.314445, VCon{F32.neg(0.786332), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.725063, VCon{F32.neg(0.065550), VCon{0.531539, VCon{F32.neg(0.138274), VCon{F32.neg(0.664977), VCon{F32.neg(0.124408), VCon{0.348485, VCon{0.411294, VCon{F32.neg(0.255674), VCon{0.560118, VCon{0.165414, VCon{F32.neg(1.080981), VCon{F32.neg(0.607011), VCon{0.525973, VCon{0.685898, VCon{F32.neg(0.678386), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.497042, VCon{F32.neg(0.043706), VCon{F32.neg(0.269620), VCon{0.375498, VCon{0.031643, VCon{0.256754, VCon{F32.neg(0.449313), VCon{F32.neg(0.236941), VCon{0.110764, VCon{F32.neg(0.506101), VCon{F32.neg(0.368964), VCon{F32.neg(0.431462), VCon{0.448110, VCon{0.459664, VCon{0.397646, VCon{F32.neg(0.147639), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.603449), VCon{0.347488, VCon{F32.neg(0.070220), VCon{0.930950, VCon{F32.neg(0.893707), VCon{F32.neg(0.117228), VCon{0.179683, VCon{F32.neg(0.197897), VCon{F32.neg(0.086651), VCon{F32.neg(0.091483), VCon{0.020928, VCon{0.757507, VCon{F32.neg(0.281873), VCon{F32.neg(0.042562), VCon{0.606788, VCon{0.043416, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.552170, VCon{F32.neg(0.054983), VCon{F32.neg(0.290567), VCon{0.352616, VCon{0.021865, VCon{0.263544, VCon{F32.neg(0.506906), VCon{F32.neg(0.224140), VCon{0.102821, VCon{F32.neg(0.521345), VCon{F32.neg(0.382453), VCon{F32.neg(0.460099), VCon{0.483263, VCon{0.433532, VCon{0.428721, VCon{F32.neg(0.165053), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.516326, VCon{F32.neg(0.020280), VCon{F32.neg(0.213495), VCon{0.377535, VCon{0.060237, VCon{0.308373, VCon{F32.neg(0.450245), VCon{F32.neg(0.227185), VCon{0.135512, VCon{F32.neg(0.524363), VCon{F32.neg(0.366344), VCon{F32.neg(0.445318), VCon{0.507188, VCon{0.480948, VCon{0.446872, VCon{F32.neg(0.138017), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.601026), VCon{0.421364, VCon{0.430481, VCon{F32.neg(0.624794), VCon{0.701813, VCon{F32.neg(0.197018), VCon{F32.neg(0.098223), VCon{F32.neg(0.671725), VCon{0.690034, VCon{0.461724, VCon{F32.neg(0.097608), VCon{F32.neg(0.656532), VCon{0.484346, VCon{F32.neg(0.521218), VCon{0.334880, VCon{F32.neg(0.368016), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.547978, VCon{F32.neg(0.028270), VCon{F32.neg(0.252550), VCon{0.388137, VCon{0.064700, VCon{0.277691, VCon{F32.neg(0.478667), VCon{F32.neg(0.245422), VCon{0.111122, VCon{F32.neg(0.503030), VCon{F32.neg(0.386521), VCon{F32.neg(0.450968), VCon{0.488073, VCon{0.445687, VCon{0.460691, VCon{F32.neg(0.149234), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.369076), VCon{0.571090, VCon{F32.neg(0.497300), VCon{F32.neg(0.271804), VCon{0.697514, VCon{F32.neg(0.209390), VCon{0.556635, VCon{0.119155, VCon{F32.neg(0.449297), VCon{F32.neg(0.092858), VCon{0.209899, VCon{0.816198, VCon{0.446426, VCon{F32.neg(0.274525), VCon{F32.neg(0.767313), VCon{0.688667, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.423245), VCon{0.049423, VCon{F32.neg(0.715018), VCon{0.352782, VCon{F32.neg(0.419563), VCon{0.779211, VCon{0.171598, VCon{0.487545, VCon{F32.neg(0.641261), VCon{F32.neg(0.350589), VCon{F32.neg(0.077344), VCon{F32.neg(0.156899), VCon{0.181278, VCon{0.542235, VCon{0.289853, VCon{F32.neg(0.340718), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.536798), VCon{0.698291, VCon{F32.neg(0.284369), VCon{F32.neg(0.020120), VCon{F32.neg(0.820187), VCon{F32.neg(0.079134), VCon{0.386410, VCon{0.450107, VCon{F32.neg(0.487505), VCon{0.614165, VCon{F32.neg(0.219134), VCon{F32.neg(0.528417), VCon{F32.neg(0.191784), VCon{0.586749, VCon{0.280185, VCon{F32.neg(0.088529), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.540734, VCon{F32.neg(0.030822), VCon{F32.neg(0.239883), VCon{0.358045, VCon{0.029604, VCon{0.296500, VCon{F32.neg(0.455411), VCon{F32.neg(0.213748), VCon{0.117175, VCon{F32.neg(0.490120), VCon{F32.neg(0.402010), VCon{F32.neg(0.473502), VCon{0.467376, VCon{0.464054, VCon{0.412708, VCon{F32.neg(0.167284), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.790142), VCon{0.076059, VCon{0.117062, VCon{0.132567, VCon{F32.neg(0.384863), VCon{0.099529, VCon{F32.neg(0.576952), VCon{0.692289, VCon{F32.neg(1.014255), VCon{F32.neg(0.238274), VCon{F32.neg(0.053503), VCon{F32.neg(0.197801), VCon{F32.neg(0.143191), VCon{0.481917, VCon{0.725843, VCon{F32.neg(0.005440), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.459331), VCon{F32.neg(0.646389), VCon{F32.neg(0.101622), VCon{0.509094, VCon{F32.neg(0.391907), VCon{F32.neg(0.465286), VCon{F32.neg(0.174023), VCon{F32.neg(0.476241), VCon{0.723181, VCon{F32.neg(0.049735), VCon{0.733048, VCon{0.803927, VCon{0.050046, VCon{F32.neg(0.085747), VCon{F32.neg(0.684722), VCon{F32.neg(0.191669), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.796539, VCon{F32.neg(0.234333), VCon{0.282990, VCon{0.458074, VCon{F32.neg(0.715959), VCon{0.031031, VCon{0.047602, VCon{F32.neg(0.646187), VCon{0.299143, VCon{F32.neg(0.559071), VCon{F32.neg(0.378227), VCon{0.417964, VCon{0.791175, VCon{F32.neg(0.293843), VCon{F32.neg(0.599317), VCon{0.346883, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.927184), VCon{0.677647, VCon{F32.neg(0.704477), VCon{F32.neg(0.293617), VCon{0.000218, VCon{0.400674, VCon{F32.neg(0.883944), VCon{F32.neg(0.038012), VCon{0.813261, VCon{0.169244, VCon{0.598483, VCon{0.701971, VCon{0.602403, VCon{F32.neg(0.094147), VCon{F32.neg(0.014421), VCon{0.004071, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.535587, VCon{F32.neg(0.007113), VCon{F32.neg(0.260145), VCon{0.361026, VCon{0.080762, VCon{0.305594, VCon{F32.neg(0.450278), VCon{F32.neg(0.281598), VCon{0.133598, VCon{F32.neg(0.506555), VCon{F32.neg(0.416464), VCon{F32.neg(0.429768), VCon{0.452217, VCon{0.464629, VCon{0.494317, VCon{F32.neg(0.165315), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.543394, VCon{F32.neg(0.036285), VCon{F32.neg(0.276159), VCon{0.346318, VCon{0.044942, VCon{0.278096, VCon{F32.neg(0.480584), VCon{F32.neg(0.241136), VCon{0.125396, VCon{F32.neg(0.524972), VCon{F32.neg(0.373807), VCon{F32.neg(0.449914), VCon{0.496610, VCon{0.468166, VCon{0.458060, VCon{F32.neg(0.155896), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.553786, VCon{F32.neg(0.029609), VCon{F32.neg(0.255964), VCon{0.370847, VCon{0.034644, VCon{0.291564, VCon{F32.neg(0.417286), VCon{F32.neg(0.230076), VCon{0.142495, VCon{F32.neg(0.513863), VCon{F32.neg(0.407865), VCon{F32.neg(0.482884), VCon{0.479628, VCon{0.476808, VCon{0.458331, VCon{F32.neg(0.130969), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.489498, VCon{F32.neg(0.051193), VCon{F32.neg(0.207873), VCon{0.388775, VCon{0.030949, VCon{0.291610, VCon{F32.neg(0.484318), VCon{F32.neg(0.189062), VCon{0.122904, VCon{F32.neg(0.511469), VCon{F32.neg(0.404553), VCon{F32.neg(0.467689), VCon{0.462194, VCon{0.453673, VCon{0.370508, VCon{F32.neg(0.157526), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.561409, VCon{F32.neg(0.024647), VCon{F32.neg(0.270024), VCon{0.364029, VCon{0.040622, VCon{0.262411, VCon{F32.neg(0.446134), VCon{F32.neg(0.254624), VCon{0.094605, VCon{F32.neg(0.507751), VCon{F32.neg(0.350535), VCon{F32.neg(0.474135), VCon{0.462553, VCon{0.489602, VCon{0.414475, VCon{F32.neg(0.178540), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.287432, VCon{F32.neg(0.712284), VCon{0.589174, VCon{F32.neg(0.868244), VCon{F32.neg(0.157593), VCon{0.710952, VCon{0.072358, VCon{F32.neg(0.610547), VCon{0.307358, VCon{F32.neg(0.337212), VCon{0.309292, VCon{0.069010, VCon{0.351116, VCon{F32.neg(0.460329), VCon{F32.neg(0.320431), VCon{0.886184, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.499293, VCon{F32.neg(0.060011), VCon{F32.neg(0.229552), VCon{0.394007, VCon{0.042161, VCon{0.315268, VCon{F32.neg(0.496220), VCon{F32.neg(0.231090), VCon{0.123797, VCon{F32.neg(0.530294), VCon{F32.neg(0.416513), VCon{F32.neg(0.486513), VCon{0.481613, VCon{0.440275, VCon{0.421603, VCon{F32.neg(0.157666), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.526003, VCon{F32.neg(0.018966), VCon{F32.neg(0.209498), VCon{0.371186, VCon{0.066981, VCon{0.299303, VCon{F32.neg(0.497291), VCon{F32.neg(0.211515), VCon{0.129340, VCon{F32.neg(0.535265), VCon{F32.neg(0.362737), VCon{F32.neg(0.464975), VCon{0.452132, VCon{0.474007, VCon{0.427060, VCon{F32.neg(0.157591), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.524002, VCon{F32.neg(0.034048), VCon{F32.neg(0.224669), VCon{0.386050, VCon{0.059118, VCon{0.294364, VCon{F32.neg(0.473671), VCon{F32.neg(0.236645), VCon{0.111179, VCon{F32.neg(0.495624), VCon{F32.neg(0.384326), VCon{F32.neg(0.472766), VCon{0.505582, VCon{0.465952, VCon{0.428683, VCon{F32.neg(0.163115), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.538797, VCon{F32.neg(0.081501), VCon{F32.neg(0.277410), VCon{0.374503, VCon{0.012380, VCon{0.289151, VCon{F32.neg(0.483823), VCon{F32.neg(0.187796), VCon{0.120549, VCon{F32.neg(0.513066), VCon{F32.neg(0.388432), VCon{F32.neg(0.479629), VCon{0.486066, VCon{0.435490, VCon{0.420002, VCon{F32.neg(0.147184), VNil{}}}}}}}}}}}}}}}}}, VLNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def param_wpe() -> VecList:
  VLCon{VCon{0.575921, VCon{0.219636, VCon{0.104740, VCon{0.182293, VCon{0.268208, VCon{0.038913, VCon{0.148183, VCon{F32.neg(0.407071), VCon{0.216663, VCon{0.065205, VCon{F32.neg(0.501253), VCon{F32.neg(0.271307), VCon{F32.neg(0.084974), VCon{F32.neg(0.441783), VCon{0.129018, VCon{F32.neg(0.255267), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.208818), VCon{0.037576, VCon{F32.neg(0.303301), VCon{0.040783, VCon{0.122208, VCon{0.151812, VCon{0.114318, VCon{0.262040, VCon{F32.neg(0.309608), VCon{F32.neg(0.407725), VCon{0.306681, VCon{0.630928, VCon{0.450263, VCon{0.224030, VCon{F32.neg(0.299707), VCon{0.194655, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.115153), VCon{0.310476, VCon{F32.neg(0.455149), VCon{0.022610, VCon{0.665252, VCon{F32.neg(0.377914), VCon{F32.neg(0.317911), VCon{0.328122, VCon{F32.neg(0.450267), VCon{0.312023, VCon{F32.neg(0.548719), VCon{F32.neg(0.254558), VCon{F32.neg(0.450855), VCon{0.524343, VCon{0.312789, VCon{0.194367, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.217601, VCon{F32.neg(0.607427), VCon{0.473008, VCon{F32.neg(0.276539), VCon{F32.neg(0.379859), VCon{F32.neg(0.072687), VCon{0.120328, VCon{F32.neg(0.195226), VCon{0.218229, VCon{F32.neg(0.011036), VCon{0.319014, VCon{0.435917, VCon{F32.neg(0.443988), VCon{F32.neg(0.307777), VCon{F32.neg(0.220326), VCon{0.312129, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.501316), VCon{0.517553, VCon{F32.neg(0.231845), VCon{0.532372, VCon{F32.neg(0.946437), VCon{0.260766, VCon{0.037400, VCon{0.504680, VCon{F32.neg(0.477446), VCon{F32.neg(0.173503), VCon{0.078879, VCon{F32.neg(0.310735), VCon{0.170238, VCon{0.488435, VCon{0.401952, VCon{F32.neg(0.245585), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.079994, VCon{0.145133, VCon{0.013169, VCon{0.264303, VCon{0.160568, VCon{F32.neg(0.412287), VCon{F32.neg(0.089013), VCon{0.226433, VCon{0.402885, VCon{0.310680, VCon{0.004824, VCon{0.204453, VCon{F32.neg(0.062917), VCon{F32.neg(0.294332), VCon{F32.neg(0.134993), VCon{F32.neg(0.445374), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.039250, VCon{F32.neg(0.314829), VCon{F32.neg(0.242137), VCon{F32.neg(0.235588), VCon{0.116914, VCon{0.412743, VCon{0.095078, VCon{0.266711, VCon{F32.neg(0.612245), VCon{F32.neg(0.378284), VCon{0.180206, VCon{0.196117, VCon{0.016362, VCon{0.287083, VCon{F32.neg(0.169076), VCon{0.211020, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.575737, VCon{0.142799, VCon{0.128699, VCon{0.429406, VCon{F32.neg(0.145415), VCon{F32.neg(0.433799), VCon{0.071950, VCon{F32.neg(0.327276), VCon{0.541682, VCon{0.269228, VCon{F32.neg(0.037512), VCon{0.155261, VCon{F32.neg(0.442948), VCon{F32.neg(0.158365), VCon{0.152583, VCon{F32.neg(0.500980), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.581321), VCon{F32.neg(0.182336), VCon{0.044814, VCon{F32.neg(0.424955), VCon{0.463995, VCon{0.116278, VCon{0.449520, VCon{0.043064, VCon{0.200684, VCon{0.275022, VCon{0.129721, VCon{F32.neg(0.762299), VCon{0.123549, VCon{0.133350, VCon{F32.neg(0.176168), VCon{0.263415, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.363290), VCon{0.464438, VCon{F32.neg(0.221461), VCon{0.309542, VCon{0.129951, VCon{F32.neg(0.273876), VCon{F32.neg(0.388567), VCon{0.195944, VCon{0.227783, VCon{0.021777, VCon{F32.neg(0.028301), VCon{F32.neg(0.085263), VCon{0.825327, VCon{F32.neg(0.143056), VCon{0.029033, VCon{F32.neg(0.472533), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.651207, VCon{0.175551, VCon{F32.neg(0.009818), VCon{0.499547, VCon{F32.neg(0.570884), VCon{0.282597, VCon{0.178560, VCon{F32.neg(0.447828), VCon{F32.neg(0.114735), VCon{F32.neg(0.490878), VCon{F32.neg(0.433615), VCon{0.321080, VCon{F32.neg(0.099795), VCon{F32.neg(0.454285), VCon{0.177599, VCon{0.172605, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.511459), VCon{F32.neg(0.029801), VCon{F32.neg(0.010139), VCon{F32.neg(0.486327), VCon{0.362088, VCon{F32.neg(0.005484), VCon{F32.neg(0.510908), VCon{F32.neg(0.053197), VCon{0.427741, VCon{0.537966, VCon{0.197780, VCon{F32.neg(0.576744), VCon{F32.neg(0.042215), VCon{F32.neg(0.017383), VCon{0.021725, VCon{0.373977, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.221633, VCon{F32.neg(0.360070), VCon{0.052745, VCon{F32.neg(0.322079), VCon{F32.neg(0.374085), VCon{0.328502, VCon{F32.neg(0.194517), VCon{F32.neg(0.103782), VCon{0.228188, VCon{F32.neg(0.375376), VCon{0.271127, VCon{0.664631, VCon{0.059022, VCon{F32.neg(0.052641), VCon{F32.neg(0.281803), VCon{0.345608, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.047869), VCon{F32.neg(0.538078), VCon{0.122666, VCon{F32.neg(0.361935), VCon{0.249915, VCon{0.306856, VCon{0.175001, VCon{0.218774, VCon{F32.neg(0.118569), VCon{F32.neg(0.273560), VCon{0.198941, VCon{0.264916, VCon{0.041351, VCon{0.000820, VCon{F32.neg(0.329949), VCon{0.408818, VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{0.013782, VCon{0.000670, VCon{F32.neg(0.005952), VCon{F32.neg(0.023393), VCon{0.030464, VCon{0.025960, VCon{0.013173, VCon{F32.neg(0.016440), VCon{F32.neg(0.002925), VCon{0.015572, VCon{0.012802, VCon{0.016145, VCon{0.011459, VCon{0.027798, VCon{F32.neg(0.004701), VCon{F32.neg(0.010769), VNil{}}}}}}}}}}}}}}}}}, VLCon{VCon{F32.neg(0.023206), VCon{F32.neg(0.029717), VCon{0.002698, VCon{0.027083, VCon{0.038994, VCon{0.021620, VCon{0.015681, VCon{0.037218, VCon{0.013451, VCon{F32.neg(0.008641), VCon{0.035997, VCon{0.026666, VCon{F32.neg(0.006367), VCon{0.022358, VCon{0.037801, VCon{0.003996, VNil{}}}}}}}}}}}}}}}}}, VLNil{}}}}}}}}}}}}}}}}}

def param_ln1_g() -> Vec:
  VCon{0.725567, VCon{0.840733, VCon{0.733895, VCon{0.933217, VCon{0.481021, VCon{1.054809, VCon{0.656914, VCon{0.615479, VCon{0.873667, VCon{0.746995, VCon{0.848092, VCon{0.852109, VCon{1.115384, VCon{0.759198, VCon{0.978086, VCon{0.815470, VNil{}}}}}}}}}}}}}}}}}

def param_ln1_b() -> Vec:
  VCon{0.211231, VCon{0.005616, VCon{0.066690, VCon{F32.neg(0.033145), VCon{F32.neg(0.009134), VCon{0.099744, VCon{F32.neg(0.214202), VCon{0.053419, VCon{F32.neg(0.193438), VCon{F32.neg(0.146479), VCon{F32.neg(0.177762), VCon{F32.neg(0.365728), VCon{F32.neg(0.020859), VCon{0.171465, VCon{0.233681, VCon{0.043295, VNil{}}}}}}}}}}}}}}}}}

def param_wq() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.011380), VCon{0.072311, VCon{F32.neg(0.078243), VCon{0.097430, VCon{0.048625, VCon{0.111853, VCon{F32.neg(0.098627), VCon{F32.neg(0.073391), VCon{0.064296, VCon{0.007693, VCon{F32.neg(0.117220), VCon{F32.neg(0.176497), VCon{0.261180, VCon{0.009062, VCon{0.069284, VCon{F32.neg(0.111596), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.063037), VCon{0.078285, VCon{0.131393, VCon{0.374896, VCon{F32.neg(0.270655), VCon{0.144401, VCon{F32.neg(0.168090), VCon{0.040048, VCon{0.196957, VCon{F32.neg(0.015850), VCon{0.048606, VCon{F32.neg(0.149510), VCon{F32.neg(0.182510), VCon{0.048646, VCon{0.402043, VCon{F32.neg(0.514592), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.206645), VCon{F32.neg(0.206352), VCon{F32.neg(0.012012), VCon{F32.neg(0.129928), VCon{0.050483, VCon{0.742811, VCon{F32.neg(0.022268), VCon{0.126959, VCon{F32.neg(0.114863), VCon{F32.neg(0.454746), VCon{0.072007, VCon{F32.neg(0.394452), VCon{0.304092, VCon{0.166976, VCon{0.214938, VCon{F32.neg(0.143052), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.269134), VCon{F32.neg(0.042424), VCon{F32.neg(0.106375), VCon{F32.neg(0.029444), VCon{F32.neg(0.034906), VCon{0.431822, VCon{F32.neg(0.326201), VCon{0.058768, VCon{0.203121, VCon{F32.neg(0.227903), VCon{F32.neg(0.003964), VCon{F32.neg(0.325285), VCon{0.386157, VCon{0.072366, VCon{0.367795, VCon{F32.neg(0.398137), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.393673), VCon{F32.neg(0.028952), VCon{0.116982, VCon{0.247441, VCon{F32.neg(0.169683), VCon{0.140674, VCon{F32.neg(0.172505), VCon{F32.neg(0.062328), VCon{0.229275, VCon{F32.neg(0.322393), VCon{0.255682, VCon{0.006462, VCon{0.271551, VCon{F32.neg(0.033547), VCon{0.127366, VCon{F32.neg(0.400408), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.236878), VCon{F32.neg(0.045565), VCon{F32.neg(0.137680), VCon{F32.neg(0.046705), VCon{0.061094, VCon{0.286791, VCon{F32.neg(0.129901), VCon{F32.neg(0.050170), VCon{0.136021, VCon{F32.neg(0.209028), VCon{F32.neg(0.057351), VCon{F32.neg(0.356310), VCon{0.448101, VCon{0.059617, VCon{0.207494, VCon{F32.neg(0.171436), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.289064, VCon{0.260102, VCon{0.011383, VCon{0.192510, VCon{0.142594, VCon{F32.neg(0.727463), VCon{0.217514, VCon{F32.neg(0.057915), VCon{0.045622, VCon{0.410012, VCon{F32.neg(0.133196), VCon{0.323964, VCon{F32.neg(0.433456), VCon{F32.neg(0.353525), VCon{F32.neg(0.370401), VCon{0.196316, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.200224, VCon{0.309418, VCon{F32.neg(0.115931), VCon{0.326094, VCon{0.039885, VCon{F32.neg(0.468145), VCon{F32.neg(0.068760), VCon{F32.neg(0.032594), VCon{0.121463, VCon{0.105993, VCon{F32.neg(0.204163), VCon{0.255156, VCon{F32.neg(0.043259), VCon{F32.neg(0.273490), VCon{F32.neg(0.258366), VCon{F32.neg(0.044105), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.103448), VCon{F32.neg(0.224335), VCon{0.181399, VCon{F32.neg(0.477759), VCon{0.009915, VCon{0.197970, VCon{0.292852, VCon{0.056680, VCon{F32.neg(0.156882), VCon{F32.neg(0.017005), VCon{0.062695, VCon{F32.neg(0.191584), VCon{F32.neg(0.373420), VCon{0.140958, VCon{0.092280, VCon{0.335790, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.036244, VCon{F32.neg(0.091401), VCon{0.185392, VCon{F32.neg(0.206827), VCon{F32.neg(0.078970), VCon{F32.neg(0.001424), VCon{0.156299, VCon{0.074949, VCon{F32.neg(0.123339), VCon{0.089132, VCon{0.051547, VCon{0.188264, VCon{F32.neg(0.518685), VCon{0.092853, VCon{F32.neg(0.062615), VCon{0.291534, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.010396, VCon{F32.neg(0.185491), VCon{0.201664, VCon{F32.neg(0.347946), VCon{F32.neg(0.015711), VCon{0.208798, VCon{0.426371, VCon{0.014096, VCon{F32.neg(0.169575), VCon{0.020884, VCon{0.052394, VCon{F32.neg(0.139370), VCon{F32.neg(0.276795), VCon{0.080037, VCon{0.018886, VCon{0.376293, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.029312), VCon{F32.neg(0.234594), VCon{0.047917, VCon{F32.neg(0.478926), VCon{F32.neg(0.012356), VCon{0.305547, VCon{0.277823, VCon{0.092580, VCon{F32.neg(0.157718), VCon{F32.neg(0.108026), VCon{0.092451, VCon{F32.neg(0.216182), VCon{F32.neg(0.237197), VCon{0.115847, VCon{0.096361, VCon{0.299306, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.340219, VCon{0.165688, VCon{F32.neg(0.082990), VCon{0.197231, VCon{0.013433, VCon{F32.neg(0.525619), VCon{0.115292, VCon{0.043575, VCon{F32.neg(0.063351), VCon{0.317942, VCon{F32.neg(0.102952), VCon{0.361922, VCon{F32.neg(0.231504), VCon{F32.neg(0.086873), VCon{F32.neg(0.330825), VCon{0.267653, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.149333, VCon{0.031927, VCon{F32.neg(0.072655), VCon{0.298110, VCon{F32.neg(0.072495), VCon{F32.neg(0.267916), VCon{F32.neg(0.256974), VCon{F32.neg(0.035126), VCon{0.209514, VCon{F32.neg(0.073307), VCon{0.076196, VCon{0.406962, VCon{0.085470, VCon{F32.neg(0.223153), VCon{F32.neg(0.427018), VCon{F32.neg(0.152129), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.015830, VCon{0.261094, VCon{0.058910, VCon{0.367113, VCon{F32.neg(0.070950), VCon{F32.neg(0.194921), VCon{0.050794, VCon{F32.neg(0.050011), VCon{0.113161, VCon{0.315296, VCon{F32.neg(0.229213), VCon{F32.neg(0.250015), VCon{F32.neg(0.408138), VCon{0.080072, VCon{0.429168, VCon{F32.neg(0.256225), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.212320), VCon{0.061285, VCon{F32.neg(0.113736), VCon{0.148790, VCon{0.036294, VCon{0.095902, VCon{F32.neg(0.225014), VCon{F32.neg(0.071538), VCon{0.163570, VCon{F32.neg(0.104826), VCon{F32.neg(0.089337), VCon{F32.neg(0.163195), VCon{0.484394, VCon{F32.neg(0.033043), VCon{0.137601, VCon{F32.neg(0.400613), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_bq() -> Vec:
  VCon{F32.neg(0.020918), VCon{F32.neg(0.100596), VCon{F32.neg(0.032050), VCon{F32.neg(0.061352), VCon{0.119403, VCon{F32.neg(0.110897), VCon{0.008693, VCon{0.060787, VCon{F32.neg(0.106087), VCon{F32.neg(0.017940), VCon{F32.neg(0.006796), VCon{F32.neg(0.041629), VCon{0.042743, VCon{0.115003, VCon{F32.neg(0.063156), VCon{F32.neg(0.006695), VNil{}}}}}}}}}}}}}}}}}

def param_wk() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.070336, VCon{0.099284, VCon{0.163225, VCon{F32.neg(0.131023), VCon{0.138901, VCon{F32.neg(0.017824), VCon{F32.neg(0.086629), VCon{0.105563, VCon{F32.neg(0.044401), VCon{0.101287, VCon{F32.neg(0.213552), VCon{F32.neg(0.288949), VCon{F32.neg(0.208730), VCon{0.009467, VCon{0.201464, VCon{F32.neg(0.059417), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.163731), VCon{0.373907, VCon{F32.neg(0.170883), VCon{0.152406, VCon{F32.neg(0.283073), VCon{F32.neg(0.005848), VCon{F32.neg(0.034090), VCon{F32.neg(0.028195), VCon{0.343600, VCon{0.161837, VCon{0.012006, VCon{0.110477, VCon{0.026605, VCon{0.051409, VCon{F32.neg(0.022189), VCon{F32.neg(0.034196), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.262401, VCon{0.316430, VCon{0.188922, VCon{F32.neg(0.042599), VCon{0.102379, VCon{0.020396, VCon{0.663873, VCon{F32.neg(0.175394), VCon{F32.neg(0.117730), VCon{0.192950, VCon{F32.neg(0.324743), VCon{F32.neg(0.574113), VCon{0.205975, VCon{F32.neg(0.220730), VCon{0.026069, VCon{F32.neg(0.036498), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.003064), VCon{0.400004, VCon{0.184328, VCon{0.028778, VCon{F32.neg(0.061564), VCon{F32.neg(0.448898), VCon{F32.neg(0.046983), VCon{0.037936, VCon{0.208401, VCon{0.457426, VCon{F32.neg(0.306874), VCon{F32.neg(0.536523), VCon{F32.neg(0.039218), VCon{F32.neg(0.075695), VCon{0.410777, VCon{F32.neg(0.143257), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.209229), VCon{0.325115, VCon{F32.neg(0.074709), VCon{0.075295, VCon{F32.neg(0.284587), VCon{F32.neg(0.312886), VCon{F32.neg(0.087030), VCon{0.153000, VCon{0.147535, VCon{0.516314, VCon{0.083385, VCon{0.109838, VCon{F32.neg(0.152731), VCon{0.000231, VCon{0.072444, VCon{F32.neg(0.066561), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.108610, VCon{0.332099, VCon{0.224692, VCon{F32.neg(0.156232), VCon{0.050914, VCon{F32.neg(0.197951), VCon{F32.neg(0.075990), VCon{0.097282, VCon{F32.neg(0.000873), VCon{0.270225, VCon{F32.neg(0.328128), VCon{F32.neg(0.529262), VCon{F32.neg(0.146875), VCon{F32.neg(0.059890), VCon{0.378415, VCon{F32.neg(0.098054), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.308901), VCon{F32.neg(0.463705), VCon{F32.neg(0.097166), VCon{F32.neg(0.060910), VCon{F32.neg(0.304663), VCon{0.175838, VCon{F32.neg(0.656978), VCon{0.270597, VCon{F32.neg(0.031070), VCon{F32.neg(0.385676), VCon{0.380181, VCon{0.663996, VCon{F32.neg(0.192772), VCon{0.227092, VCon{F32.neg(0.116033), VCon{0.074078, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.230339), VCon{F32.neg(0.248113), VCon{0.065284, VCon{F32.neg(0.149717), VCon{F32.neg(0.307420), VCon{F32.neg(0.038992), VCon{F32.neg(0.516329), VCon{0.299349, VCon{0.064695, VCon{F32.neg(0.045370), VCon{0.163675, VCon{0.187958, VCon{F32.neg(0.498162), VCon{0.157410, VCon{0.080585, VCon{0.023439, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.236516, VCon{0.045681, VCon{F32.neg(0.090961), VCon{0.161513, VCon{0.338384, VCon{0.223368, VCon{0.316895, VCon{F32.neg(0.145006), VCon{F32.neg(0.015449), VCon{F32.neg(0.282417), VCon{F32.neg(0.165177), VCon{0.011228, VCon{0.629052, VCon{F32.neg(0.153818), VCon{F32.neg(0.099979), VCon{0.083098, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.023179), VCon{F32.neg(0.075373), VCon{F32.neg(0.221045), VCon{0.145705, VCon{0.123326, VCon{0.156102, VCon{0.339927, VCon{F32.neg(0.188424), VCon{0.015755, VCon{F32.neg(0.293736), VCon{0.124425, VCon{0.255436, VCon{0.418807, VCon{F32.neg(0.014350), VCon{F32.neg(0.207155), VCon{0.034506, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.178560, VCon{F32.neg(0.086925), VCon{F32.neg(0.050183), VCon{0.101730, VCon{0.210242, VCon{0.196901, VCon{0.302351, VCon{F32.neg(0.118642), VCon{F32.neg(0.110111), VCon{F32.neg(0.264721), VCon{F32.neg(0.073218), VCon{0.022045, VCon{0.536240, VCon{F32.neg(0.052963), VCon{F32.neg(0.216352), VCon{F32.neg(0.008340), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.247149, VCon{F32.neg(0.034687), VCon{F32.neg(0.045684), VCon{0.033331, VCon{0.299568, VCon{0.180135, VCon{0.377979, VCon{F32.neg(0.121799), VCon{0.017241, VCon{F32.neg(0.237231), VCon{F32.neg(0.137037), VCon{F32.neg(0.165014), VCon{0.631927, VCon{F32.neg(0.136834), VCon{F32.neg(0.136617), VCon{0.039805, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.115397), VCon{F32.neg(0.353388), VCon{F32.neg(0.109910), VCon{F32.neg(0.064665), VCon{0.047885, VCon{0.083062, VCon{F32.neg(0.076243), VCon{0.157640, VCon{F32.neg(0.051251), VCon{F32.neg(0.389860), VCon{0.305198, VCon{0.524030, VCon{F32.neg(0.170084), VCon{0.117868, VCon{F32.neg(0.230710), VCon{0.100233, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.364378), VCon{F32.neg(0.008699), VCon{F32.neg(0.136403), VCon{0.013504, VCon{F32.neg(0.327851), VCon{F32.neg(0.449437), VCon{F32.neg(0.055329), VCon{0.087579, VCon{0.160008, VCon{0.371204, VCon{0.309683, VCon{0.340663, VCon{F32.neg(0.260573), VCon{0.221790, VCon{F32.neg(0.019494), VCon{F32.neg(0.021884), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.093772, VCon{0.039852, VCon{F32.neg(0.041935), VCon{0.091562, VCon{0.002633, VCon{0.337260, VCon{F32.neg(0.341326), VCon{F32.neg(0.058086), VCon{0.001195, VCon{F32.neg(0.362023), VCon{F32.neg(0.039529), VCon{0.061626, VCon{F32.neg(0.031136), VCon{F32.neg(0.077861), VCon{0.195939, VCon{F32.neg(0.113861), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.029269), VCon{0.183672, VCon{0.231102, VCon{F32.neg(0.043343), VCon{F32.neg(0.171734), VCon{F32.neg(0.257957), VCon{F32.neg(0.418613), VCon{0.175024, VCon{0.072346, VCon{0.407550, VCon{F32.neg(0.167278), VCon{F32.neg(0.156697), VCon{F32.neg(0.320324), VCon{F32.neg(0.002663), VCon{0.276000, VCon{F32.neg(0.101033), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_bk() -> Vec:
  VCon{F32.neg(0.000587), VCon{F32.neg(0.007373), VCon{F32.neg(0.006133), VCon{F32.neg(0.010312), VCon{F32.neg(0.007764), VCon{F32.neg(0.008454), VCon{0.003100, VCon{F32.neg(0.001336), VCon{0.001610, VCon{0.001279, VCon{0.019058, VCon{0.010178, VCon{0.006712, VCon{F32.neg(0.002723), VCon{0.001353, VCon{F32.neg(0.010348), VNil{}}}}}}}}}}}}}}}}}

def param_wv() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.113495, VCon{0.067243, VCon{0.012812, VCon{0.037013, VCon{0.027708, VCon{F32.neg(0.099586), VCon{F32.neg(0.028208), VCon{F32.neg(0.046262), VCon{0.109842, VCon{F32.neg(0.028867), VCon{F32.neg(0.140304), VCon{0.048704, VCon{0.011954, VCon{0.036801, VCon{F32.neg(0.002476), VCon{0.074463, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.085065), VCon{F32.neg(0.054177), VCon{F32.neg(0.048354), VCon{0.025125, VCon{0.060664, VCon{0.118737, VCon{F32.neg(0.000553), VCon{0.039220, VCon{F32.neg(0.099612), VCon{F32.neg(0.023788), VCon{0.026753, VCon{0.113659, VCon{F32.neg(0.108977), VCon{F32.neg(0.012531), VCon{F32.neg(0.007661), VCon{F32.neg(0.134729), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.011277, VCon{0.087860, VCon{0.040235, VCon{F32.neg(0.030239), VCon{0.119471, VCon{F32.neg(0.009952), VCon{0.066822, VCon{0.046763, VCon{F32.neg(0.414246), VCon{F32.neg(0.004472), VCon{F32.neg(0.148125), VCon{0.019514, VCon{F32.neg(0.042419), VCon{F32.neg(0.085703), VCon{F32.neg(0.032542), VCon{0.055036, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.041563), VCon{F32.neg(0.069857), VCon{0.246387, VCon{0.071886, VCon{F32.neg(0.054451), VCon{F32.neg(0.380738), VCon{F32.neg(0.035807), VCon{F32.neg(0.075733), VCon{0.238463, VCon{0.001928, VCon{0.124916, VCon{0.077477, VCon{0.049061, VCon{0.067908, VCon{F32.neg(0.157356), VCon{0.032393, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.063076, VCon{0.052612, VCon{0.029591, VCon{0.042392, VCon{0.079740, VCon{0.093667, VCon{0.084150, VCon{0.010448, VCon{F32.neg(0.305622), VCon{0.003455, VCon{F32.neg(0.158891), VCon{F32.neg(0.059350), VCon{F32.neg(0.062222), VCon{F32.neg(0.087638), VCon{0.098706, VCon{F32.neg(0.120415), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.151716), VCon{0.058861, VCon{0.128523, VCon{0.075244, VCon{F32.neg(0.185170), VCon{F32.neg(0.131341), VCon{F32.neg(0.035049), VCon{0.038971, VCon{0.139473, VCon{0.104129, VCon{F32.neg(0.028330), VCon{F32.neg(0.023474), VCon{0.009463, VCon{0.088853, VCon{0.001468, VCon{0.076666, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.318543, VCon{F32.neg(0.150750), VCon{0.335017, VCon{0.235522, VCon{F32.neg(0.059852), VCon{F32.neg(0.377492), VCon{F32.neg(0.065663), VCon{F32.neg(0.193867), VCon{0.178837, VCon{0.136017, VCon{F32.neg(0.152869), VCon{0.119579, VCon{F32.neg(0.157263), VCon{F32.neg(0.056870), VCon{0.096558, VCon{F32.neg(0.051760), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.037570), VCon{0.128921, VCon{F32.neg(0.074029), VCon{0.012723, VCon{0.118218, VCon{F32.neg(0.024958), VCon{0.007594, VCon{0.043453, VCon{F32.neg(0.292766), VCon{F32.neg(0.049831), VCon{F32.neg(0.091759), VCon{F32.neg(0.041509), VCon{0.016971, VCon{F32.neg(0.004038), VCon{F32.neg(0.012140), VCon{0.108432, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.077447, VCon{0.015802, VCon{0.213293, VCon{0.310741, VCon{F32.neg(0.058886), VCon{0.080433, VCon{0.042091, VCon{F32.neg(0.183546), VCon{0.057395, VCon{F32.neg(0.068531), VCon{F32.neg(0.001097), VCon{F32.neg(0.040307), VCon{0.132012, VCon{F32.neg(0.131471), VCon{F32.neg(0.003458), VCon{F32.neg(0.175616), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.079065, VCon{F32.neg(0.052307), VCon{F32.neg(0.091074), VCon{0.030717, VCon{F32.neg(0.088551), VCon{0.037360, VCon{F32.neg(0.050259), VCon{F32.neg(0.053749), VCon{0.141697, VCon{F32.neg(0.086087), VCon{0.034050, VCon{0.013094, VCon{0.228601, VCon{0.054652, VCon{F32.neg(0.157530), VCon{0.071284, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.152000), VCon{0.133569, VCon{F32.neg(0.316513), VCon{F32.neg(0.351528), VCon{F32.neg(0.020404), VCon{0.028807, VCon{F32.neg(0.017309), VCon{0.116726, VCon{F32.neg(0.154046), VCon{0.179357, VCon{F32.neg(0.027616), VCon{F32.neg(0.000881), VCon{0.018396, VCon{0.150797, VCon{F32.neg(0.024314), VCon{0.271653, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.080962), VCon{0.205264, VCon{F32.neg(0.174178), VCon{F32.neg(0.237619), VCon{0.001202, VCon{0.102316, VCon{0.048824, VCon{0.144036, VCon{F32.neg(0.274347), VCon{0.075984, VCon{F32.neg(0.047367), VCon{F32.neg(0.064425), VCon{F32.neg(0.035629), VCon{0.055233, VCon{0.029250, VCon{F32.neg(0.070860), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.157749, VCon{F32.neg(0.100527), VCon{0.012292, VCon{0.117626, VCon{F32.neg(0.035495), VCon{0.110813, VCon{0.012718, VCon{F32.neg(0.066987), VCon{0.184523, VCon{0.015098, VCon{F32.neg(0.000111), VCon{0.007575, VCon{0.004009, VCon{F32.neg(0.050965), VCon{0.002301, VCon{F32.neg(0.095010), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.017388), VCon{F32.neg(0.028166), VCon{0.163659, VCon{0.031423, VCon{F32.neg(0.054150), VCon{F32.neg(0.106069), VCon{0.010813, VCon{F32.neg(0.049452), VCon{0.210055, VCon{0.039943, VCon{0.081965, VCon{F32.neg(0.045126), VCon{0.107781, VCon{0.010254, VCon{F32.neg(0.016147), VCon{0.083976, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.055712, VCon{F32.neg(0.006238), VCon{0.212144, VCon{0.050060, VCon{0.093413, VCon{F32.neg(0.170326), VCon{F32.neg(0.026630), VCon{F32.neg(0.078095), VCon{0.103586, VCon{0.130459, VCon{F32.neg(0.105719), VCon{0.103934, VCon{F32.neg(0.271627), VCon{F32.neg(0.064885), VCon{0.188483, VCon{F32.neg(0.149386), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.033539, VCon{0.057162, VCon{0.040208, VCon{0.164953, VCon{F32.neg(0.099216), VCon{0.052144, VCon{F32.neg(0.036093), VCon{F32.neg(0.135614), VCon{0.120432, VCon{F32.neg(0.019661), VCon{0.014968, VCon{F32.neg(0.068356), VCon{F32.neg(0.052316), VCon{F32.neg(0.058539), VCon{0.159891, VCon{F32.neg(0.208036), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_bv() -> Vec:
  VCon{F32.neg(0.069975), VCon{0.075762, VCon{0.076719, VCon{F32.neg(0.093515), VCon{0.017509, VCon{0.001725, VCon{F32.neg(0.093185), VCon{F32.neg(0.000125), VCon{F32.neg(0.053165), VCon{F32.neg(0.129058), VCon{0.065205, VCon{0.083081, VCon{0.025199, VCon{F32.neg(0.080004), VCon{F32.neg(0.073173), VCon{F32.neg(0.029465), VNil{}}}}}}}}}}}}}}}}}

def param_w_attn_proj() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.071926, VCon{F32.neg(0.005508), VCon{0.087907, VCon{0.129421, VCon{F32.neg(0.084174), VCon{F32.neg(0.077420), VCon{0.059591, VCon{F32.neg(0.063256), VCon{F32.neg(0.125077), VCon{F32.neg(0.186348), VCon{0.050027, VCon{0.051282, VCon{F32.neg(0.165657), VCon{F32.neg(0.017462), VCon{0.149861, VCon{F32.neg(0.154209), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.111632, VCon{F32.neg(0.119593), VCon{0.066835, VCon{0.042615, VCon{0.073544, VCon{F32.neg(0.033097), VCon{0.078498, VCon{0.062073, VCon{0.157245, VCon{0.053618, VCon{F32.neg(0.124204), VCon{F32.neg(0.102506), VCon{0.098995, VCon{0.098423, VCon{0.072167, VCon{0.045290, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.144750, VCon{F32.neg(0.077482), VCon{F32.neg(0.081148), VCon{0.126952, VCon{F32.neg(0.253277), VCon{F32.neg(0.011789), VCon{F32.neg(0.194826), VCon{F32.neg(0.011563), VCon{F32.neg(0.238011), VCon{F32.neg(0.037042), VCon{0.137730, VCon{F32.neg(0.084119), VCon{F32.neg(0.148172), VCon{0.076479, VCon{0.156847, VCon{F32.neg(0.201882), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.001867, VCon{0.110203, VCon{F32.neg(0.005531), VCon{0.032360, VCon{F32.neg(0.033377), VCon{F32.neg(0.059236), VCon{0.147429, VCon{F32.neg(0.082611), VCon{F32.neg(0.099939), VCon{F32.neg(0.031823), VCon{0.061337, VCon{0.016328, VCon{0.036827, VCon{F32.neg(0.136886), VCon{0.029294, VCon{F32.neg(0.124051), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.006578), VCon{0.055018, VCon{F32.neg(0.022319), VCon{F32.neg(0.138795), VCon{0.154234, VCon{0.121189, VCon{0.083004, VCon{0.036030, VCon{F32.neg(0.086841), VCon{F32.neg(0.266976), VCon{0.039721, VCon{0.098318, VCon{F32.neg(0.047606), VCon{F32.neg(0.076011), VCon{0.241600, VCon{0.091428, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.024001), VCon{F32.neg(0.023040), VCon{0.208993, VCon{F32.neg(0.109202), VCon{0.191512, VCon{F32.neg(0.121881), VCon{F32.neg(0.052584), VCon{0.078289, VCon{0.169324, VCon{0.031183, VCon{F32.neg(0.114828), VCon{0.034326, VCon{0.069152, VCon{0.010463, VCon{F32.neg(0.069486), VCon{0.080253, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.181013, VCon{F32.neg(0.064951), VCon{0.197102, VCon{F32.neg(0.091022), VCon{0.210137, VCon{F32.neg(0.094718), VCon{0.036585, VCon{0.270000, VCon{0.215264, VCon{0.149151, VCon{F32.neg(0.194740), VCon{F32.neg(0.096954), VCon{0.027719, VCon{0.066530, VCon{0.134332, VCon{0.200769, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.108318), VCon{0.076240, VCon{F32.neg(0.077610), VCon{0.018096, VCon{F32.neg(0.135856), VCon{0.058380, VCon{0.031237, VCon{F32.neg(0.127656), VCon{0.117124, VCon{0.076264, VCon{F32.neg(0.004917), VCon{0.001016, VCon{0.133051, VCon{F32.neg(0.023338), VCon{F32.neg(0.163162), VCon{0.109560, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.044793), VCon{F32.neg(0.014241), VCon{F32.neg(0.088298), VCon{F32.neg(0.017329), VCon{F32.neg(0.033670), VCon{0.108783, VCon{0.163223, VCon{F32.neg(0.090196), VCon{F32.neg(0.198826), VCon{F32.neg(0.183784), VCon{0.082130, VCon{0.039298, VCon{F32.neg(0.037955), VCon{F32.neg(0.034383), VCon{0.249972, VCon{F32.neg(0.037843), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.110129, VCon{F32.neg(0.213442), VCon{0.002182, VCon{F32.neg(0.026634), VCon{0.059213, VCon{0.085931, VCon{0.090522, VCon{0.079032, VCon{F32.neg(0.024855), VCon{0.141474, VCon{F32.neg(0.055738), VCon{F32.neg(0.073984), VCon{F32.neg(0.175012), VCon{0.081940, VCon{0.262601, VCon{0.038034, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.050678), VCon{F32.neg(0.036742), VCon{F32.neg(0.108440), VCon{F32.neg(0.071043), VCon{0.030522, VCon{0.017342, VCon{0.024202, VCon{F32.neg(0.082679), VCon{0.153867, VCon{0.103317, VCon{F32.neg(0.165042), VCon{F32.neg(0.140029), VCon{0.099177, VCon{0.110265, VCon{F32.neg(0.062596), VCon{0.128752, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.038471), VCon{F32.neg(0.088970), VCon{F32.neg(0.148408), VCon{0.293810, VCon{F32.neg(0.125334), VCon{0.170808, VCon{0.040959, VCon{F32.neg(0.100209), VCon{0.146840, VCon{0.242113, VCon{F32.neg(0.179537), VCon{F32.neg(0.232566), VCon{F32.neg(0.025645), VCon{0.256676, VCon{0.050323, VCon{0.094726, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.005348), VCon{0.001132, VCon{0.021419, VCon{0.058028, VCon{F32.neg(0.048792), VCon{F32.neg(0.097834), VCon{0.041803, VCon{F32.neg(0.109001), VCon{0.018932, VCon{F32.neg(0.042614), VCon{F32.neg(0.140991), VCon{F32.neg(0.162389), VCon{0.049378, VCon{F32.neg(0.017938), VCon{F32.neg(0.033495), VCon{F32.neg(0.002485), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.068583), VCon{0.085432, VCon{0.064956, VCon{F32.neg(0.089562), VCon{F32.neg(0.123526), VCon{F32.neg(0.085598), VCon{F32.neg(0.159262), VCon{0.101839, VCon{F32.neg(0.094621), VCon{0.025400, VCon{0.141577, VCon{0.105036, VCon{F32.neg(0.006850), VCon{F32.neg(0.092141), VCon{F32.neg(0.312004), VCon{F32.neg(0.160964), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.064365), VCon{0.065248, VCon{0.060941, VCon{0.082843, VCon{F32.neg(0.161728), VCon{0.037911, VCon{F32.neg(0.140118), VCon{F32.neg(0.040734), VCon{F32.neg(0.205179), VCon{F32.neg(0.182711), VCon{0.268484, VCon{0.225949, VCon{F32.neg(0.082928), VCon{F32.neg(0.123608), VCon{F32.neg(0.340491), VCon{F32.neg(0.204435), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.047280, VCon{F32.neg(0.011800), VCon{0.107072, VCon{F32.neg(0.039140), VCon{0.102024, VCon{F32.neg(0.170956), VCon{F32.neg(0.188935), VCon{0.125829, VCon{0.027555, VCon{0.063839, VCon{F32.neg(0.029683), VCon{F32.neg(0.000678), VCon{F32.neg(0.129922), VCon{0.065174, VCon{F32.neg(0.120426), VCon{F32.neg(0.027420), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_b_attn_proj() -> Vec:
  VCon{F32.neg(0.073581), VCon{0.041939, VCon{F32.neg(0.057302), VCon{F32.neg(0.042888), VCon{0.036389, VCon{F32.neg(0.101844), VCon{0.150581, VCon{F32.neg(0.024811), VCon{0.057215, VCon{0.088228, VCon{0.111037, VCon{0.104111, VCon{0.011701, VCon{F32.neg(0.096942), VCon{F32.neg(0.165593), VCon{0.025246, VNil{}}}}}}}}}}}}}}}}}

def param_ln2_g() -> Vec:
  VCon{0.628523, VCon{1.249192, VCon{1.024529, VCon{0.861884, VCon{1.054402, VCon{1.280286, VCon{1.028428, VCon{1.000011, VCon{1.050193, VCon{1.053760, VCon{1.080876, VCon{1.100259, VCon{0.995407, VCon{1.003916, VCon{0.904676, VCon{1.365909, VNil{}}}}}}}}}}}}}}}}}

def param_ln2_b() -> Vec:
  VCon{0.142109, VCon{0.022316, VCon{F32.neg(0.132521), VCon{0.053952, VCon{0.091285, VCon{0.038410, VCon{F32.neg(0.148688), VCon{F32.neg(0.084987), VCon{0.081462, VCon{0.011978, VCon{0.012709, VCon{F32.neg(0.041074), VCon{F32.neg(0.054920), VCon{F32.neg(0.013704), VCon{0.126187, VCon{F32.neg(0.076241), VNil{}}}}}}}}}}}}}}}}}

def param_w_mlp_fc() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.055066), VCon{F32.neg(0.414801), VCon{0.049601, VCon{F32.neg(0.195098), VCon{0.143724, VCon{0.224682, VCon{0.229168, VCon{0.032690, VCon{F32.neg(0.034688), VCon{F32.neg(0.253727), VCon{0.108461, VCon{F32.neg(0.043363), VCon{0.209570, VCon{0.102829, VCon{F32.neg(0.071198), VCon{0.442955, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.024504), VCon{F32.neg(0.359431), VCon{F32.neg(0.097256), VCon{F32.neg(0.172695), VCon{0.134310, VCon{0.524562, VCon{0.065162, VCon{0.046970, VCon{0.149256, VCon{F32.neg(0.248841), VCon{0.095849, VCon{F32.neg(0.072821), VCon{0.198673, VCon{0.112586, VCon{F32.neg(0.090711), VCon{0.106597, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.316901), VCon{0.245175, VCon{F32.neg(0.212469), VCon{0.280962, VCon{0.207637, VCon{F32.neg(0.803797), VCon{F32.neg(0.003892), VCon{0.546806, VCon{0.017200, VCon{0.046487, VCon{0.006004, VCon{0.447423, VCon{0.012131, VCon{0.329576, VCon{0.002312, VCon{F32.neg(0.660391), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.156336, VCon{F32.neg(0.448131), VCon{0.131365, VCon{F32.neg(0.209113), VCon{0.228668, VCon{0.193962, VCon{F32.neg(0.096324), VCon{F32.neg(0.119010), VCon{0.071422, VCon{F32.neg(0.161937), VCon{F32.neg(0.001618), VCon{F32.neg(0.015326), VCon{F32.neg(0.043139), VCon{0.200875, VCon{0.044052, VCon{0.430863, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.079039, VCon{0.023259, VCon{0.183087, VCon{F32.neg(0.244062), VCon{0.249331, VCon{F32.neg(0.127794), VCon{0.133144, VCon{0.322704, VCon{F32.neg(0.093936), VCon{0.092224, VCon{F32.neg(0.181258), VCon{F32.neg(0.213312), VCon{F32.neg(0.042729), VCon{F32.neg(0.217113), VCon{0.040863, VCon{F32.neg(0.237536), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.125656), VCon{F32.neg(0.007713), VCon{F32.neg(0.398769), VCon{F32.neg(0.063450), VCon{0.123138, VCon{0.577245, VCon{F32.neg(0.043173), VCon{0.450144, VCon{F32.neg(0.033498), VCon{F32.neg(0.295548), VCon{0.143581, VCon{0.022140, VCon{0.392547, VCon{0.040920, VCon{F32.neg(0.144783), VCon{F32.neg(0.065763), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.267249), VCon{0.062923, VCon{F32.neg(0.247040), VCon{F32.neg(0.096508), VCon{0.316371, VCon{0.062408, VCon{F32.neg(0.185008), VCon{0.005249, VCon{0.143626, VCon{0.268621, VCon{0.073061, VCon{F32.neg(0.362620), VCon{0.404616, VCon{F32.neg(0.009335), VCon{F32.neg(0.214730), VCon{0.261870, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.192964, VCon{F32.neg(0.376152), VCon{0.147667, VCon{0.344071, VCon{F32.neg(0.658452), VCon{0.101756, VCon{0.051028, VCon{F32.neg(0.079435), VCon{0.065279, VCon{0.071994, VCon{0.188205, VCon{F32.neg(0.138523), VCon{F32.neg(0.336871), VCon{0.299358, VCon{0.190782, VCon{F32.neg(0.150358), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.376033, VCon{F32.neg(0.304855), VCon{F32.neg(0.057127), VCon{F32.neg(0.252938), VCon{0.065665, VCon{0.376942, VCon{F32.neg(0.270269), VCon{F32.neg(0.118185), VCon{0.081606, VCon{F32.neg(0.025840), VCon{F32.neg(0.190954), VCon{F32.neg(0.121362), VCon{0.072891, VCon{0.070546, VCon{0.034549, VCon{0.495422, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.168134), VCon{0.043366, VCon{0.040564, VCon{0.234506, VCon{F32.neg(0.161152), VCon{F32.neg(0.051673), VCon{F32.neg(0.081645), VCon{0.180283, VCon{F32.neg(0.353687), VCon{F32.neg(0.103734), VCon{0.032177, VCon{F32.neg(0.246350), VCon{0.300091, VCon{0.083980, VCon{F32.neg(0.023980), VCon{0.045561, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.136716, VCon{F32.neg(0.339692), VCon{0.230939, VCon{F32.neg(0.265847), VCon{0.176455, VCon{0.096614, VCon{0.017245, VCon{0.015944, VCon{F32.neg(0.150640), VCon{F32.neg(0.180380), VCon{F32.neg(0.135526), VCon{0.057631, VCon{F32.neg(0.103419), VCon{0.231901, VCon{0.036248, VCon{0.426250, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.200366, VCon{F32.neg(0.341500), VCon{0.149022, VCon{F32.neg(0.311849), VCon{0.096317, VCon{0.162482, VCon{F32.neg(0.200351), VCon{F32.neg(0.119504), VCon{0.076969, VCon{F32.neg(0.291128), VCon{F32.neg(0.028596), VCon{F32.neg(0.092274), VCon{0.015297, VCon{0.221393, VCon{0.019689, VCon{0.496810, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.106967, VCon{0.142008, VCon{0.171433, VCon{0.051890, VCon{0.192354, VCon{0.079187, VCon{0.080927, VCon{F32.neg(0.211013), VCon{F32.neg(0.274609), VCon{F32.neg(0.128190), VCon{F32.neg(0.413066), VCon{F32.neg(0.185440), VCon{F32.neg(0.083593), VCon{F32.neg(0.079586), VCon{0.147240, VCon{F32.neg(0.020739), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.250563, VCon{0.496492, VCon{0.224286, VCon{0.238897, VCon{F32.neg(0.539423), VCon{F32.neg(0.282538), VCon{F32.neg(0.177475), VCon{0.408003, VCon{F32.neg(0.166658), VCon{0.028428, VCon{F32.neg(0.190330), VCon{F32.neg(0.050933), VCon{F32.neg(0.142836), VCon{0.131384, VCon{0.115903, VCon{F32.neg(0.081126), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.002547, VCon{0.005933, VCon{0.156477, VCon{0.055439, VCon{0.181210, VCon{0.100482, VCon{0.143754, VCon{0.263014, VCon{F32.neg(0.158680), VCon{F32.neg(0.176750), VCon{F32.neg(0.234104), VCon{F32.neg(0.313810), VCon{0.074695, VCon{F32.neg(0.123781), VCon{0.131395, VCon{F32.neg(0.260361), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.006015, VCon{0.043611, VCon{F32.neg(0.062902), VCon{0.177679, VCon{F32.neg(0.100165), VCon{0.023437, VCon{F32.neg(0.516371), VCon{F32.neg(0.070075), VCon{0.156096, VCon{F32.neg(0.171974), VCon{F32.neg(0.042326), VCon{F32.neg(0.223995), VCon{F32.neg(0.180060), VCon{0.166245, VCon{0.326137, VCon{F32.neg(0.140149), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.506070), VCon{0.470522, VCon{F32.neg(0.252852), VCon{F32.neg(0.091362), VCon{F32.neg(0.063170), VCon{0.450831, VCon{0.052970, VCon{0.427961, VCon{F32.neg(0.591724), VCon{0.007384, VCon{0.266233, VCon{F32.neg(0.081851), VCon{0.523485, VCon{0.323721, VCon{F32.neg(0.080172), VCon{F32.neg(0.030650), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.114063), VCon{0.150123, VCon{F32.neg(0.347880), VCon{F32.neg(0.026796), VCon{F32.neg(0.022786), VCon{0.127253, VCon{F32.neg(0.408595), VCon{0.047671, VCon{F32.neg(0.013909), VCon{0.114254, VCon{0.064187, VCon{F32.neg(0.008579), VCon{0.390934, VCon{F32.neg(0.022639), VCon{F32.neg(0.001652), VCon{0.194121, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.475095, VCon{F32.neg(0.116021), VCon{0.112454, VCon{0.204387, VCon{F32.neg(0.313548), VCon{0.198087, VCon{0.181712, VCon{0.080312, VCon{F32.neg(0.142760), VCon{F32.neg(0.149011), VCon{F32.neg(0.116663), VCon{0.120476, VCon{0.011144, VCon{F32.neg(0.012544), VCon{F32.neg(0.190472), VCon{0.014226, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.051503, VCon{F32.neg(0.145235), VCon{0.012881, VCon{0.006442, VCon{F32.neg(0.295589), VCon{0.305495, VCon{F32.neg(0.180210), VCon{F32.neg(0.403940), VCon{0.261326, VCon{F32.neg(0.200466), VCon{0.155476, VCon{0.145118, VCon{0.079310, VCon{F32.neg(0.084637), VCon{0.159826, VCon{F32.neg(0.011048), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.066526, VCon{0.078120, VCon{0.423573, VCon{F32.neg(0.189613), VCon{0.195966, VCon{F32.neg(0.045249), VCon{0.301893, VCon{F32.neg(0.238764), VCon{F32.neg(0.154953), VCon{0.071069, VCon{F32.neg(0.313568), VCon{F32.neg(0.250818), VCon{F32.neg(0.240978), VCon{F32.neg(0.156918), VCon{0.101286, VCon{F32.neg(0.027958), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.018443), VCon{F32.neg(0.012072), VCon{0.154697, VCon{0.320758, VCon{F32.neg(0.087374), VCon{F32.neg(0.076742), VCon{0.058421, VCon{F32.neg(0.040890), VCon{F32.neg(0.156664), VCon{F32.neg(0.153740), VCon{F32.neg(0.082556), VCon{F32.neg(0.380340), VCon{0.130080, VCon{0.091548, VCon{0.081360, VCon{F32.neg(0.202538), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.056102, VCon{0.138340, VCon{F32.neg(0.128036), VCon{F32.neg(0.152110), VCon{F32.neg(0.225957), VCon{0.245553, VCon{0.157904, VCon{0.060250, VCon{F32.neg(0.239581), VCon{0.170086, VCon{F32.neg(0.007249), VCon{F32.neg(0.180441), VCon{F32.neg(0.002173), VCon{0.157374, VCon{0.031594, VCon{0.147037, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.194456, VCon{F32.neg(0.117706), VCon{0.102289, VCon{F32.neg(0.095104), VCon{0.234418, VCon{0.084920, VCon{0.140942, VCon{F32.neg(0.040714), VCon{0.045719, VCon{0.377804, VCon{F32.neg(0.258834), VCon{F32.neg(0.504957), VCon{F32.neg(0.327714), VCon{F32.neg(0.162830), VCon{0.118831, VCon{F32.neg(0.306636), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.054088), VCon{F32.neg(0.315651), VCon{F32.neg(0.311229), VCon{F32.neg(0.234445), VCon{F32.neg(0.044218), VCon{0.486419, VCon{0.232863, VCon{F32.neg(0.062735), VCon{0.025183, VCon{F32.neg(0.412964), VCon{0.292177, VCon{0.187389, VCon{0.224518, VCon{0.103050, VCon{F32.neg(0.279269), VCon{F32.neg(0.046509), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.231783), VCon{0.035484, VCon{F32.neg(0.598054), VCon{F32.neg(0.108156), VCon{0.424134, VCon{F32.neg(0.015255), VCon{F32.neg(0.525122), VCon{0.146104, VCon{0.397665, VCon{0.160907, VCon{0.179450, VCon{F32.neg(0.076716), VCon{0.158705, VCon{0.124056, VCon{F32.neg(0.040008), VCon{F32.neg(0.166089), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.321765, VCon{F32.neg(0.071618), VCon{0.009362, VCon{F32.neg(0.217572), VCon{0.272415, VCon{F32.neg(0.123958), VCon{F32.neg(0.170170), VCon{0.042723, VCon{F32.neg(0.263315), VCon{0.236652, VCon{F32.neg(0.453075), VCon{F32.neg(0.383156), VCon{F32.neg(0.066137), VCon{0.247566, VCon{F32.neg(0.003005), VCon{0.232617, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.161162, VCon{F32.neg(0.081922), VCon{0.197355, VCon{0.074629, VCon{F32.neg(0.184364), VCon{0.119270, VCon{F32.neg(0.059626), VCon{0.263649, VCon{0.015353, VCon{0.054913, VCon{F32.neg(0.062057), VCon{F32.neg(0.250374), VCon{F32.neg(0.211477), VCon{0.230197, VCon{0.261851, VCon{F32.neg(0.340562), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.303311), VCon{0.106668, VCon{0.023142, VCon{F32.neg(0.154205), VCon{F32.neg(0.403732), VCon{0.093796, VCon{F32.neg(0.088180), VCon{0.508786, VCon{F32.neg(0.502427), VCon{0.010461, VCon{F32.neg(0.023179), VCon{F32.neg(0.075874), VCon{F32.neg(0.107640), VCon{0.472264, VCon{0.306520, VCon{F32.neg(0.050265), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.153138), VCon{F32.neg(0.419120), VCon{0.183767, VCon{F32.neg(0.174442), VCon{0.158924, VCon{0.146347, VCon{0.282838, VCon{F32.neg(0.140209), VCon{F32.neg(0.079635), VCon{F32.neg(0.589132), VCon{0.226657, VCon{0.057971, VCon{0.283747, VCon{0.114683, VCon{F32.neg(0.215746), VCon{0.323384, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.262081, VCon{F32.neg(0.055153), VCon{0.075912, VCon{0.204711, VCon{0.122449, VCon{0.044442, VCon{F32.neg(0.031215), VCon{F32.neg(0.229745), VCon{0.385562, VCon{F32.neg(0.112076), VCon{F32.neg(0.140875), VCon{F32.neg(0.267560), VCon{F32.neg(0.177078), VCon{F32.neg(0.028250), VCon{0.095362, VCon{F32.neg(0.286433), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.390721), VCon{0.160233, VCon{0.118387, VCon{0.014312, VCon{F32.neg(0.100228), VCon{0.356319, VCon{F32.neg(0.186549), VCon{0.020875, VCon{F32.neg(0.051677), VCon{F32.neg(0.194489), VCon{F32.neg(0.039810), VCon{F32.neg(0.006894), VCon{0.304917, VCon{0.032942, VCon{0.223837, VCon{F32.neg(0.171402), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.109580, VCon{F32.neg(0.248450), VCon{0.165463, VCon{F32.neg(0.219920), VCon{0.046279, VCon{0.134024, VCon{F32.neg(0.030223), VCon{F32.neg(0.198692), VCon{F32.neg(0.208277), VCon{F32.neg(0.156073), VCon{0.007751, VCon{F32.neg(0.041339), VCon{F32.neg(0.019273), VCon{0.229263, VCon{0.006135, VCon{0.539703, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.189538), VCon{F32.neg(0.332956), VCon{F32.neg(0.297716), VCon{F32.neg(0.058841), VCon{0.023867, VCon{0.211158, VCon{F32.neg(0.036796), VCon{0.108350, VCon{0.212779, VCon{0.048207, VCon{0.175777, VCon{0.286832, VCon{F32.neg(0.090005), VCon{0.164441, VCon{0.068229, VCon{F32.neg(0.417051), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.029946, VCon{F32.neg(0.048783), VCon{F32.neg(0.078090), VCon{F32.neg(0.022548), VCon{F32.neg(0.019734), VCon{0.105859, VCon{0.029675, VCon{0.004271, VCon{F32.neg(0.091877), VCon{F32.neg(0.097470), VCon{F32.neg(0.028463), VCon{0.006454, VCon{0.081919, VCon{F32.neg(0.044047), VCon{F32.neg(0.001409), VCon{0.058942, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.301880, VCon{0.067405, VCon{F32.neg(0.111992), VCon{0.021390, VCon{0.512133, VCon{F32.neg(0.131182), VCon{F32.neg(0.158537), VCon{0.176147, VCon{F32.neg(0.400678), VCon{0.058378, VCon{F32.neg(0.268901), VCon{F32.neg(0.138011), VCon{F32.neg(0.214057), VCon{0.093019, VCon{F32.neg(0.122398), VCon{0.286907, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.130020, VCon{F32.neg(0.073255), VCon{0.365792, VCon{0.375257, VCon{0.004765, VCon{F32.neg(0.553597), VCon{F32.neg(0.238165), VCon{0.308187, VCon{0.106195, VCon{F32.neg(0.059968), VCon{F32.neg(0.255503), VCon{F32.neg(0.022114), VCon{0.111321, VCon{F32.neg(0.047311), VCon{0.072095, VCon{F32.neg(0.376849), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.063759), VCon{F32.neg(0.416047), VCon{F32.neg(0.204056), VCon{F32.neg(0.165465), VCon{0.069295, VCon{0.335393, VCon{0.262385, VCon{F32.neg(0.047849), VCon{0.112195, VCon{F32.neg(0.246112), VCon{0.199446, VCon{0.011080, VCon{0.147416, VCon{0.154504, VCon{F32.neg(0.045939), VCon{F32.neg(0.002364), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.355976, VCon{F32.neg(0.177439), VCon{0.101493, VCon{0.181803, VCon{0.059213, VCon{F32.neg(0.283934), VCon{0.553573, VCon{F32.neg(0.064999), VCon{F32.neg(0.005460), VCon{F32.neg(0.012110), VCon{0.096189, VCon{0.361013, VCon{F32.neg(0.302575), VCon{F32.neg(0.281857), VCon{F32.neg(0.305270), VCon{F32.neg(0.455841), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.215715), VCon{0.120903, VCon{F32.neg(0.328271), VCon{F32.neg(0.034152), VCon{0.419069, VCon{0.088796, VCon{F32.neg(0.241348), VCon{0.113329, VCon{F32.neg(0.012668), VCon{0.062320, VCon{F32.neg(0.083583), VCon{F32.neg(0.131169), VCon{F32.neg(0.199849), VCon{0.176463, VCon{0.140700, VCon{0.000366, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.457270), VCon{0.036849, VCon{0.115022, VCon{F32.neg(0.473287), VCon{0.037402, VCon{0.188793, VCon{F32.neg(0.124635), VCon{F32.neg(0.294238), VCon{0.406154, VCon{0.260349, VCon{F32.neg(0.015932), VCon{F32.neg(0.307140), VCon{F32.neg(0.105162), VCon{F32.neg(0.001104), VCon{0.281516, VCon{0.298835, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.132413), VCon{0.368503, VCon{F32.neg(0.073439), VCon{0.387077, VCon{F32.neg(0.152459), VCon{F32.neg(0.427831), VCon{F32.neg(0.307130), VCon{0.006503, VCon{F32.neg(0.181412), VCon{F32.neg(0.311175), VCon{F32.neg(0.118896), VCon{0.121969, VCon{0.233172, VCon{0.218351, VCon{0.086613, VCon{F32.neg(0.210669), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.149307, VCon{F32.neg(0.135805), VCon{0.309471, VCon{0.227730, VCon{F32.neg(0.427806), VCon{0.368384, VCon{F32.neg(0.247619), VCon{F32.neg(0.028682), VCon{F32.neg(0.007709), VCon{0.024967, VCon{F32.neg(0.134457), VCon{F32.neg(0.072722), VCon{F32.neg(0.614080), VCon{0.319043, VCon{0.436584, VCon{F32.neg(0.134046), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.068781), VCon{0.380311, VCon{F32.neg(0.008378), VCon{F32.neg(0.100613), VCon{0.138380, VCon{F32.neg(0.281922), VCon{F32.neg(0.057032), VCon{F32.neg(0.323645), VCon{0.321257, VCon{0.185622, VCon{F32.neg(0.082476), VCon{0.505370, VCon{F32.neg(0.237423), VCon{F32.neg(0.329879), VCon{0.051479, VCon{F32.neg(0.397566), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.066456, VCon{F32.neg(0.299767), VCon{F32.neg(0.038674), VCon{F32.neg(0.162222), VCon{0.149290, VCon{0.189354, VCon{F32.neg(0.014842), VCon{F32.neg(0.015415), VCon{0.048459, VCon{F32.neg(0.272166), VCon{0.073673, VCon{F32.neg(0.092885), VCon{0.215605, VCon{0.167539, VCon{F32.neg(0.014927), VCon{0.497380, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.421016, VCon{F32.neg(0.039712), VCon{0.120101, VCon{0.327327, VCon{0.035747, VCon{0.212747, VCon{0.160134, VCon{F32.neg(0.347551), VCon{0.155076, VCon{F32.neg(0.311563), VCon{F32.neg(0.362833), VCon{0.036360, VCon{0.057971, VCon{F32.neg(0.310076), VCon{0.152916, VCon{F32.neg(0.382778), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.038284), VCon{0.309421, VCon{0.030660, VCon{0.282404, VCon{F32.neg(0.081794), VCon{F32.neg(0.193661), VCon{F32.neg(0.095856), VCon{0.214684, VCon{F32.neg(0.164441), VCon{0.017083, VCon{0.007074, VCon{0.111365, VCon{F32.neg(0.058020), VCon{F32.neg(0.169684), VCon{F32.neg(0.060115), VCon{F32.neg(0.077633), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.150416), VCon{0.108921, VCon{F32.neg(0.104066), VCon{F32.neg(0.172035), VCon{F32.neg(0.292814), VCon{0.116644, VCon{F32.neg(0.409000), VCon{F32.neg(0.199982), VCon{0.254761, VCon{0.045947, VCon{0.148625, VCon{0.137549, VCon{0.136169, VCon{F32.neg(0.052481), VCon{0.134545, VCon{0.201910, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.034309, VCon{F32.neg(0.255458), VCon{F32.neg(0.019686), VCon{0.058748, VCon{F32.neg(0.403439), VCon{0.292186, VCon{0.298753, VCon{0.122570, VCon{F32.neg(0.556927), VCon{0.141118, VCon{0.142009, VCon{0.150228, VCon{F32.neg(0.355458), VCon{0.471427, VCon{0.049041, VCon{0.187864, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.100362), VCon{F32.neg(0.375611), VCon{F32.neg(0.195666), VCon{F32.neg(0.137504), VCon{0.226434, VCon{0.458465, VCon{0.003448, VCon{0.110398, VCon{0.046916, VCon{F32.neg(0.188127), VCon{0.095674, VCon{F32.neg(0.028282), VCon{0.062335, VCon{0.200616, VCon{F32.neg(0.056456), VCon{0.133815, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.233260, VCon{F32.neg(0.262159), VCon{0.364524, VCon{0.162610, VCon{0.209093, VCon{F32.neg(0.256741), VCon{F32.neg(0.155686), VCon{0.376329, VCon{0.198734, VCon{0.071638, VCon{F32.neg(0.194352), VCon{F32.neg(0.415981), VCon{F32.neg(0.025784), VCon{F32.neg(0.082981), VCon{0.025067, VCon{F32.neg(0.344282), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.242643), VCon{0.109062, VCon{F32.neg(0.089375), VCon{F32.neg(0.110413), VCon{F32.neg(0.071547), VCon{0.002601, VCon{F32.neg(0.413332), VCon{F32.neg(0.298131), VCon{0.389807, VCon{0.041059, VCon{0.101151, VCon{0.309010, VCon{0.049359, VCon{0.007452, VCon{0.244993, VCon{F32.neg(0.256507), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.039613, VCon{0.048011, VCon{0.061487, VCon{F32.neg(0.026846), VCon{F32.neg(0.309170), VCon{0.095659, VCon{F32.neg(0.071523), VCon{0.029746, VCon{F32.neg(0.109738), VCon{F32.neg(0.096380), VCon{0.118178, VCon{F32.neg(0.106911), VCon{0.031859, VCon{0.038268, VCon{0.089659, VCon{F32.neg(0.113072), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.182173), VCon{F32.neg(0.425185), VCon{F32.neg(0.313053), VCon{F32.neg(0.347041), VCon{F32.neg(0.126858), VCon{0.554042, VCon{0.564671, VCon{0.242038, VCon{F32.neg(0.074606), VCon{F32.neg(0.287576), VCon{0.452673, VCon{F32.neg(0.187878), VCon{0.330424, VCon{0.205601, VCon{F32.neg(0.304344), VCon{0.211450, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.234452, VCon{0.016863, VCon{0.413718, VCon{0.087521, VCon{0.268065, VCon{0.356727, VCon{0.446322, VCon{F32.neg(0.610958), VCon{F32.neg(0.295132), VCon{F32.neg(0.632684), VCon{F32.neg(0.632538), VCon{F32.neg(0.325890), VCon{0.126215, VCon{F32.neg(0.112317), VCon{0.442624, VCon{F32.neg(0.091971), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.102860), VCon{0.173630, VCon{F32.neg(0.260362), VCon{0.076649, VCon{F32.neg(0.125507), VCon{0.230097, VCon{F32.neg(0.062511), VCon{0.091313, VCon{F32.neg(0.429452), VCon{0.177721, VCon{F32.neg(0.073280), VCon{F32.neg(0.070846), VCon{F32.neg(0.123160), VCon{0.374330, VCon{0.332680, VCon{F32.neg(0.055469), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.083726, VCon{0.078131, VCon{0.072946, VCon{0.026495, VCon{F32.neg(0.006446), VCon{0.255350, VCon{F32.neg(0.047482), VCon{F32.neg(0.176723), VCon{F32.neg(0.488880), VCon{F32.neg(0.212966), VCon{F32.neg(0.091573), VCon{F32.neg(0.018044), VCon{0.133095, VCon{0.071405, VCon{0.112949, VCon{0.383963, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.388378, VCon{F32.neg(0.053461), VCon{0.163029, VCon{F32.neg(0.087782), VCon{F32.neg(0.109362), VCon{F32.neg(0.003390), VCon{F32.neg(0.103631), VCon{0.160229, VCon{F32.neg(0.138352), VCon{0.367377, VCon{F32.neg(0.166144), VCon{F32.neg(0.040032), VCon{F32.neg(0.297705), VCon{F32.neg(0.185367), VCon{0.120594, VCon{F32.neg(0.185010), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.121024), VCon{F32.neg(0.384644), VCon{0.118495, VCon{0.174475, VCon{F32.neg(0.177133), VCon{F32.neg(0.111388), VCon{0.208975, VCon{F32.neg(0.048646), VCon{0.080070, VCon{0.092377, VCon{0.191639, VCon{F32.neg(0.019128), VCon{F32.neg(0.002765), VCon{0.149476, VCon{F32.neg(0.031400), VCon{F32.neg(0.466062), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.009550), VCon{F32.neg(0.253321), VCon{F32.neg(0.195640), VCon{F32.neg(0.035918), VCon{0.107536, VCon{0.117430, VCon{0.219228, VCon{0.050535, VCon{0.064926, VCon{F32.neg(0.122250), VCon{0.193506, VCon{F32.neg(0.442774), VCon{0.282944, VCon{0.064532, VCon{F32.neg(0.172661), VCon{F32.neg(0.088110), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.061094), VCon{0.225144, VCon{F32.neg(0.434872), VCon{0.229673, VCon{F32.neg(0.260252), VCon{0.183234, VCon{F32.neg(0.266609), VCon{0.034903, VCon{0.087110, VCon{F32.neg(0.211144), VCon{0.198109, VCon{0.293264, VCon{0.630157, VCon{0.137735, VCon{0.001870, VCon{F32.neg(0.272271), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.111119), VCon{F32.neg(0.108049), VCon{F32.neg(0.270207), VCon{F32.neg(0.217273), VCon{0.190913, VCon{0.558514, VCon{F32.neg(0.014175), VCon{0.619533, VCon{F32.neg(0.296333), VCon{F32.neg(0.005205), VCon{0.096853, VCon{F32.neg(0.048054), VCon{0.286293, VCon{0.099198, VCon{F32.neg(0.158268), VCon{F32.neg(0.352654), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.005713, VCon{F32.neg(0.012875), VCon{0.062136, VCon{F32.neg(0.203956), VCon{0.232554, VCon{0.117929, VCon{0.132188, VCon{F32.neg(0.091227), VCon{F32.neg(0.015204), VCon{F32.neg(0.066666), VCon{F32.neg(0.217909), VCon{F32.neg(0.178319), VCon{F32.neg(0.131732), VCon{F32.neg(0.122951), VCon{0.160778, VCon{F32.neg(0.190683), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.109688), VCon{0.041062, VCon{F32.neg(0.076236), VCon{F32.neg(0.243170), VCon{0.197878, VCon{F32.neg(0.009812), VCon{F32.neg(0.314453), VCon{0.029178, VCon{0.166510, VCon{0.156350, VCon{0.098403, VCon{F32.neg(0.369615), VCon{F32.neg(0.038940), VCon{0.016897, VCon{0.034477, VCon{0.085435, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }
  }
  }

def param_b_mlp_fc() -> Vec:
  VCon{F32.neg(0.065053), VCon{F32.neg(0.137222), VCon{F32.neg(0.256841), VCon{F32.neg(0.135437), VCon{F32.neg(0.104830), VCon{F32.neg(0.093903), VCon{F32.neg(0.039926), VCon{0.034333, VCon{F32.neg(0.143300), VCon{F32.neg(0.060931), VCon{F32.neg(0.101732), VCon{F32.neg(0.159944), VCon{F32.neg(0.067411), VCon{F32.neg(0.203527), VCon{F32.neg(0.145670), VCon{F32.neg(0.221514), VCon{F32.neg(0.113559), VCon{F32.neg(0.028873), VCon{F32.neg(0.086914), VCon{0.030584, VCon{F32.neg(0.073196), VCon{F32.neg(0.023274), VCon{F32.neg(0.019040), VCon{0.024531, VCon{F32.neg(0.182476), VCon{0.024372, VCon{F32.neg(0.229235), VCon{F32.neg(0.218491), VCon{F32.neg(0.285102), VCon{F32.neg(0.296958), VCon{F32.neg(0.173361), VCon{F32.neg(0.058152), VCon{F32.neg(0.086636), VCon{F32.neg(0.095234), VCon{0.013305, VCon{F32.neg(0.114782), VCon{F32.neg(0.177510), VCon{F32.neg(0.126245), VCon{F32.neg(0.177607), VCon{F32.neg(0.152609), VCon{0.008630, VCon{F32.neg(0.227272), VCon{F32.neg(0.290453), VCon{F32.neg(0.146630), VCon{F32.neg(0.088258), VCon{F32.neg(0.097173), VCon{F32.neg(0.068352), VCon{F32.neg(0.012838), VCon{F32.neg(0.145790), VCon{F32.neg(0.242917), VCon{F32.neg(0.182247), VCon{0.011698, VCon{F32.neg(0.146943), VCon{F32.neg(0.018750), VCon{F32.neg(0.212614), VCon{F32.neg(0.119235), VCon{F32.neg(0.133723), VCon{0.016403, VCon{F32.neg(0.251042), VCon{0.018000, VCon{F32.neg(0.018089), VCon{F32.neg(0.148317), VCon{F32.neg(0.113439), VCon{F32.neg(0.112924), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def param_w_mlp_proj() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.109449, VCon{F32.neg(0.038323), VCon{0.072524, VCon{0.273440, VCon{F32.neg(0.151193), VCon{F32.neg(0.127041), VCon{F32.neg(0.366044), VCon{0.034904, VCon{F32.neg(0.066458), VCon{F32.neg(0.424806), VCon{0.275376, VCon{0.149571, VCon{0.156907, VCon{F32.neg(0.558529), VCon{0.043950, VCon{F32.neg(0.218103), VCon{F32.neg(0.358016), VCon{F32.neg(0.142321), VCon{0.415490, VCon{0.284476, VCon{0.247173, VCon{F32.neg(0.005697), VCon{F32.neg(0.356698), VCon{0.171849, VCon{0.091752, VCon{F32.neg(0.277227), VCon{F32.neg(0.035027), VCon{F32.neg(0.095295), VCon{F32.neg(0.333563), VCon{0.222307, VCon{F32.neg(0.104253), VCon{0.196831, VCon{0.018938, VCon{0.249820, VCon{F32.neg(0.012005), VCon{F32.neg(0.186818), VCon{F32.neg(0.315765), VCon{0.228680, VCon{0.748355, VCon{F32.neg(0.095025), VCon{0.192893, VCon{F32.neg(0.100435), VCon{F32.neg(0.176351), VCon{0.259216, VCon{0.092030, VCon{0.164865, VCon{F32.neg(0.157076), VCon{0.086533, VCon{0.250755, VCon{0.037319, VCon{F32.neg(0.184769), VCon{0.131668, VCon{F32.neg(0.156883), VCon{F32.neg(0.549292), VCon{0.662243, VCon{F32.neg(0.017903), VCon{0.007862, VCon{0.358198, VCon{0.188720, VCon{F32.neg(0.426076), VCon{0.282829, VCon{F32.neg(0.159102), VCon{0.085528, VCon{F32.neg(0.014259), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.128269), VCon{F32.neg(0.007760), VCon{F32.neg(0.506291), VCon{F32.neg(0.484884), VCon{0.126965, VCon{0.150567, VCon{0.186419, VCon{0.279399, VCon{F32.neg(0.379846), VCon{0.192753, VCon{F32.neg(0.428874), VCon{F32.neg(0.465660), VCon{0.120394, VCon{F32.neg(0.090015), VCon{0.173753, VCon{0.019531, VCon{0.406083, VCon{0.143360, VCon{0.101502, VCon{F32.neg(0.151393), VCon{0.282608, VCon{0.219059, VCon{0.045780, VCon{0.468031, VCon{F32.neg(0.020883), VCon{F32.neg(0.401563), VCon{F32.neg(0.272597), VCon{0.240024, VCon{0.147679, VCon{0.000713, VCon{0.197647, VCon{F32.neg(0.065932), VCon{F32.neg(0.387164), VCon{F32.neg(0.009895), VCon{0.154824, VCon{F32.neg(0.256913), VCon{0.215890, VCon{F32.neg(0.036880), VCon{F32.neg(0.107258), VCon{0.040573, VCon{F32.neg(0.083113), VCon{F32.neg(0.003368), VCon{0.189689, VCon{F32.neg(0.446134), VCon{F32.neg(0.268116), VCon{0.304374, VCon{F32.neg(0.170056), VCon{F32.neg(0.126390), VCon{F32.neg(0.125811), VCon{F32.neg(0.109239), VCon{0.302050, VCon{F32.neg(0.323062), VCon{0.184331, VCon{0.386313, VCon{0.234283, VCon{F32.neg(0.073512), VCon{F32.neg(0.177351), VCon{0.166992, VCon{0.159895, VCon{0.209368, VCon{0.087042, VCon{0.357248, VCon{0.121951, VCon{0.046406, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.101371, VCon{0.133705, VCon{0.171554, VCon{0.299697, VCon{F32.neg(0.028717), VCon{0.225825, VCon{0.106208, VCon{0.142101, VCon{0.127773, VCon{F32.neg(0.409617), VCon{0.070969, VCon{0.301593, VCon{F32.neg(0.308086), VCon{F32.neg(0.246721), VCon{F32.neg(0.213052), VCon{0.050153, VCon{F32.neg(0.480324), VCon{0.069955, VCon{F32.neg(0.098080), VCon{0.152401, VCon{F32.neg(0.576999), VCon{F32.neg(0.177190), VCon{F32.neg(0.075953), VCon{0.085551, VCon{0.151083, VCon{0.532672, VCon{F32.neg(0.242571), VCon{F32.neg(0.010192), VCon{F32.neg(0.125974), VCon{0.234582, VCon{F32.neg(0.023176), VCon{F32.neg(0.063802), VCon{0.124197, VCon{0.126410, VCon{0.047717, VCon{F32.neg(0.108642), VCon{F32.neg(0.072923), VCon{0.237821, VCon{0.011610, VCon{0.096209, VCon{0.139272, VCon{F32.neg(0.237608), VCon{0.036006, VCon{F32.neg(0.198489), VCon{0.215438, VCon{F32.neg(0.098288), VCon{F32.neg(0.097788), VCon{0.063087, VCon{0.124923, VCon{0.038757, VCon{0.199829, VCon{F32.neg(0.004459), VCon{F32.neg(0.042226), VCon{F32.neg(0.167493), VCon{F32.neg(0.492382), VCon{F32.neg(0.137567), VCon{0.023149, VCon{0.153624, VCon{0.006937, VCon{0.130100, VCon{0.134101, VCon{F32.neg(0.401222), VCon{0.018386, VCon{0.172009, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.001119, VCon{F32.neg(0.169014), VCon{0.029671, VCon{F32.neg(0.429902), VCon{0.032685, VCon{F32.neg(0.341257), VCon{F32.neg(0.506753), VCon{0.031452, VCon{F32.neg(0.279363), VCon{F32.neg(0.083647), VCon{F32.neg(0.172693), VCon{F32.neg(0.439564), VCon{0.292517, VCon{0.293023, VCon{F32.neg(0.143809), VCon{F32.neg(0.229753), VCon{0.230408, VCon{F32.neg(0.207791), VCon{0.304275, VCon{F32.neg(0.138778), VCon{0.346849, VCon{F32.neg(0.043827), VCon{0.337972, VCon{F32.neg(0.030568), VCon{0.109651, VCon{F32.neg(0.521863), VCon{F32.neg(0.092721), VCon{F32.neg(0.069371), VCon{0.315848, VCon{F32.neg(0.087696), VCon{0.006551, VCon{F32.neg(0.010726), VCon{F32.neg(0.042452), VCon{F32.neg(0.030197), VCon{F32.neg(0.084081), VCon{0.006246, VCon{F32.neg(0.071973), VCon{F32.neg(0.082973), VCon{0.477173, VCon{F32.neg(0.070715), VCon{F32.neg(0.497052), VCon{F32.neg(0.086853), VCon{F32.neg(0.054038), VCon{0.649701, VCon{F32.neg(0.240108), VCon{0.195678, VCon{0.099352, VCon{F32.neg(0.297127), VCon{0.372918, VCon{0.024845, VCon{F32.neg(0.375206), VCon{F32.neg(0.002748), VCon{F32.neg(0.127852), VCon{0.407672, VCon{0.294999, VCon{0.231709, VCon{0.280241, VCon{0.178627, VCon{F32.neg(0.061973), VCon{F32.neg(0.182541), VCon{F32.neg(0.448185), VCon{F32.neg(0.128898), VCon{0.115766, VCon{F32.neg(0.366992), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.141689), VCon{0.091263, VCon{0.248995, VCon{0.150168, VCon{0.043261, VCon{F32.neg(0.020887), VCon{0.449864, VCon{F32.neg(0.096742), VCon{0.188451, VCon{F32.neg(0.146420), VCon{F32.neg(0.013260), VCon{F32.neg(0.001142), VCon{0.184849, VCon{F32.neg(0.243633), VCon{F32.neg(0.130693), VCon{0.113982, VCon{F32.neg(0.265926), VCon{0.153667, VCon{F32.neg(0.337390), VCon{F32.neg(0.165722), VCon{0.279545, VCon{F32.neg(0.099656), VCon{F32.neg(0.364334), VCon{0.083944, VCon{0.221652, VCon{0.169828, VCon{0.242641, VCon{F32.neg(0.232980), VCon{F32.neg(0.276794), VCon{F32.neg(0.458266), VCon{F32.neg(0.040091), VCon{F32.neg(0.051639), VCon{F32.neg(0.017071), VCon{0.384602, VCon{F32.neg(0.050726), VCon{0.226785, VCon{0.192379, VCon{F32.neg(0.176435), VCon{F32.neg(0.133796), VCon{0.257141, VCon{F32.neg(0.025223), VCon{F32.neg(0.199510), VCon{F32.neg(0.123741), VCon{F32.neg(0.204653), VCon{F32.neg(0.165365), VCon{F32.neg(0.075642), VCon{F32.neg(0.077612), VCon{F32.neg(0.007757), VCon{F32.neg(0.380562), VCon{0.016794, VCon{0.274943, VCon{F32.neg(0.045458), VCon{F32.neg(0.187783), VCon{F32.neg(0.372943), VCon{0.064674, VCon{F32.neg(0.132460), VCon{F32.neg(0.068874), VCon{0.049425, VCon{0.231964, VCon{F32.neg(0.122181), VCon{0.080407, VCon{F32.neg(0.115547), VCon{0.077878, VCon{0.028995, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.103933, VCon{F32.neg(0.133561), VCon{F32.neg(0.390521), VCon{0.021172, VCon{F32.neg(0.194255), VCon{0.104620, VCon{F32.neg(0.189207), VCon{0.032661, VCon{F32.neg(0.215661), VCon{0.415083, VCon{0.122967, VCon{0.180473, VCon{F32.neg(0.153583), VCon{0.060400, VCon{0.161491, VCon{0.194249, VCon{0.504071, VCon{0.013624, VCon{F32.neg(0.005582), VCon{0.503071, VCon{0.040078, VCon{0.253903, VCon{0.065365, VCon{F32.neg(0.220221), VCon{F32.neg(0.279039), VCon{F32.neg(0.101901), VCon{F32.neg(0.435573), VCon{0.018884, VCon{0.270728, VCon{0.209343, VCon{F32.neg(0.106909), VCon{0.477352, VCon{0.139051, VCon{F32.neg(0.240697), VCon{F32.neg(0.090950), VCon{F32.neg(0.178433), VCon{F32.neg(0.328489), VCon{F32.neg(0.081904), VCon{F32.neg(0.251037), VCon{F32.neg(0.273898), VCon{0.535400, VCon{0.233809, VCon{0.177954, VCon{F32.neg(0.010348), VCon{0.131578, VCon{0.082217, VCon{0.180693, VCon{0.526740, VCon{0.260948, VCon{F32.neg(0.105762), VCon{F32.neg(0.256848), VCon{0.183341, VCon{0.180785, VCon{F32.neg(0.243149), VCon{0.089404, VCon{0.133809, VCon{F32.neg(0.050969), VCon{F32.neg(0.214582), VCon{F32.neg(0.290259), VCon{F32.neg(0.166910), VCon{0.199535, VCon{0.209027, VCon{F32.neg(0.042123), VCon{0.127173, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.163134, VCon{0.356291, VCon{F32.neg(0.402466), VCon{0.076598, VCon{F32.neg(0.181581), VCon{0.255881, VCon{F32.neg(0.273746), VCon{0.150789, VCon{0.006369, VCon{F32.neg(0.224153), VCon{0.103314, VCon{0.122095, VCon{0.143874, VCon{F32.neg(0.087949), VCon{F32.neg(0.037884), VCon{0.228263, VCon{0.025160, VCon{F32.neg(0.243328), VCon{0.116181, VCon{F32.neg(0.037399), VCon{F32.neg(0.002611), VCon{0.072646, VCon{0.049932, VCon{0.070335, VCon{0.345346, VCon{F32.neg(0.153515), VCon{F32.neg(0.146885), VCon{F32.neg(0.119981), VCon{F32.neg(0.392983), VCon{0.170703, VCon{0.191264, VCon{F32.neg(0.036452), VCon{0.069628, VCon{F32.neg(0.025621), VCon{F32.neg(0.001348), VCon{0.067661, VCon{F32.neg(0.133695), VCon{0.205979, VCon{F32.neg(0.155677), VCon{F32.neg(0.308754), VCon{0.087672, VCon{F32.neg(0.205571), VCon{0.119171, VCon{0.345491, VCon{0.026174, VCon{0.270388, VCon{F32.neg(0.049108), VCon{0.052324, VCon{0.001421, VCon{0.206994, VCon{F32.neg(0.067392), VCon{0.111839, VCon{0.053367, VCon{0.631546, VCon{0.349307, VCon{F32.neg(0.355872), VCon{0.339636, VCon{F32.neg(0.014196), VCon{F32.neg(0.019393), VCon{0.133743, VCon{F32.neg(0.128429), VCon{0.120833, VCon{0.139640, VCon{F32.neg(0.051980), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.211558, VCon{0.271958, VCon{0.221850, VCon{0.177665, VCon{F32.neg(0.149416), VCon{F32.neg(0.062752), VCon{0.016867, VCon{0.213023, VCon{0.107482, VCon{0.093286, VCon{0.002669, VCon{0.103243, VCon{F32.neg(0.007971), VCon{0.003765, VCon{0.008192, VCon{0.187210, VCon{F32.neg(0.057682), VCon{F32.neg(0.300253), VCon{F32.neg(0.131559), VCon{F32.neg(0.067693), VCon{F32.neg(0.221218), VCon{0.033377, VCon{F32.neg(0.093370), VCon{F32.neg(0.245702), VCon{0.310007, VCon{F32.neg(0.149549), VCon{F32.neg(0.028647), VCon{0.000408, VCon{0.064177, VCon{0.183427, VCon{0.020343, VCon{F32.neg(0.148756), VCon{0.145132, VCon{0.262676, VCon{F32.neg(0.134032), VCon{0.027819, VCon{0.003609, VCon{0.475036, VCon{0.029077, VCon{0.078993, VCon{F32.neg(0.007761), VCon{0.049911, VCon{0.075776, VCon{F32.neg(0.317194), VCon{F32.neg(0.013135), VCon{0.052138, VCon{F32.neg(0.053353), VCon{F32.neg(0.038710), VCon{0.133426, VCon{0.148104, VCon{0.014991, VCon{F32.neg(0.267028), VCon{0.034921, VCon{0.504403, VCon{0.170495, VCon{0.019236, VCon{0.100375, VCon{F32.neg(0.156316), VCon{0.252206, VCon{0.096881, VCon{F32.neg(0.145387), VCon{F32.neg(0.049737), VCon{F32.neg(0.142269), VCon{F32.neg(0.007823), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.052750), VCon{F32.neg(0.259165), VCon{0.325486, VCon{0.048818, VCon{0.498010, VCon{F32.neg(0.186469), VCon{0.111986, VCon{F32.neg(0.380507), VCon{0.036441, VCon{0.231500, VCon{0.190005, VCon{0.054956, VCon{0.107783, VCon{0.067771, VCon{0.265175, VCon{F32.neg(0.179107), VCon{F32.neg(0.047850), VCon{0.058272, VCon{0.026275, VCon{F32.neg(0.115018), VCon{0.269894, VCon{0.194488, VCon{F32.neg(0.036424), VCon{0.033798, VCon{F32.neg(0.384963), VCon{0.056238, VCon{0.227444, VCon{F32.neg(0.153589), VCon{F32.neg(0.044457), VCon{F32.neg(0.174990), VCon{F32.neg(0.110370), VCon{0.162158, VCon{0.031556, VCon{F32.neg(0.275935), VCon{F32.neg(0.131667), VCon{0.175943, VCon{0.234584, VCon{F32.neg(0.381908), VCon{F32.neg(0.042512), VCon{F32.neg(0.091173), VCon{F32.neg(0.090423), VCon{0.168864, VCon{F32.neg(0.250667), VCon{0.154082, VCon{F32.neg(0.096621), VCon{F32.neg(0.035534), VCon{0.294769, VCon{F32.neg(0.024881), VCon{F32.neg(0.272043), VCon{F32.neg(0.277025), VCon{0.226680, VCon{F32.neg(0.179358), VCon{F32.neg(0.113963), VCon{F32.neg(0.715724), VCon{0.074203, VCon{F32.neg(0.104554), VCon{F32.neg(0.030273), VCon{F32.neg(0.042167), VCon{F32.neg(0.139151), VCon{F32.neg(0.102919), VCon{F32.neg(0.138011), VCon{F32.neg(0.019046), VCon{0.041134, VCon{0.003174, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.360254, VCon{0.198913, VCon{0.251191, VCon{0.107442, VCon{0.311507, VCon{F32.neg(0.083781), VCon{F32.neg(0.179939), VCon{F32.neg(0.081262), VCon{0.127572, VCon{F32.neg(0.009427), VCon{0.162457, VCon{0.116046, VCon{0.065408, VCon{F32.neg(0.075892), VCon{0.416607, VCon{0.104969, VCon{F32.neg(0.615703), VCon{F32.neg(0.116382), VCon{F32.neg(0.081608), VCon{F32.neg(0.252229), VCon{0.060908, VCon{0.287400, VCon{F32.neg(0.205251), VCon{0.255358, VCon{0.289305, VCon{F32.neg(0.066155), VCon{0.007031, VCon{0.124433, VCon{F32.neg(0.144714), VCon{0.432813, VCon{0.295314, VCon{F32.neg(0.111346), VCon{F32.neg(0.005332), VCon{F32.neg(0.168485), VCon{0.118215, VCon{0.003205, VCon{0.555497, VCon{0.329892, VCon{F32.neg(0.313931), VCon{F32.neg(0.141543), VCon{0.008112, VCon{0.048869, VCon{F32.neg(0.201247), VCon{F32.neg(0.448895), VCon{0.169064, VCon{0.343159, VCon{F32.neg(0.104654), VCon{F32.neg(0.172176), VCon{F32.neg(0.160691), VCon{0.128394, VCon{0.638461, VCon{F32.neg(0.420985), VCon{0.043646, VCon{0.402132, VCon{0.135309, VCon{F32.neg(0.161104), VCon{F32.neg(0.004265), VCon{F32.neg(0.143951), VCon{0.165028, VCon{0.512899, VCon{F32.neg(0.123573), VCon{F32.neg(0.081329), VCon{0.047654, VCon{F32.neg(0.139764), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.105309, VCon{0.154504, VCon{0.521281, VCon{0.325783, VCon{0.134730, VCon{0.036536, VCon{0.112588, VCon{F32.neg(0.383898), VCon{0.316902, VCon{F32.neg(0.029146), VCon{0.295152, VCon{0.362688, VCon{F32.neg(0.216330), VCon{0.182572, VCon{0.133660, VCon{0.067116, VCon{F32.neg(0.195044), VCon{F32.neg(0.006045), VCon{F32.neg(0.301875), VCon{F32.neg(0.412438), VCon{F32.neg(0.126351), VCon{F32.neg(0.205425), VCon{F32.neg(0.008254), VCon{F32.neg(0.385462), VCon{F32.neg(0.012515), VCon{0.296094, VCon{0.146176, VCon{0.051542, VCon{F32.neg(0.360709), VCon{0.089604, VCon{0.059849, VCon{F32.neg(0.082935), VCon{0.427021, VCon{0.122038, VCon{F32.neg(0.155235), VCon{F32.neg(0.000983), VCon{0.328684, VCon{0.114596, VCon{F32.neg(0.235535), VCon{0.098787, VCon{F32.neg(0.077496), VCon{0.443826, VCon{F32.neg(0.047194), VCon{0.005143, VCon{0.088247, VCon{F32.neg(0.380304), VCon{0.239371, VCon{0.041014, VCon{F32.neg(0.176051), VCon{0.109027, VCon{0.049463, VCon{F32.neg(0.101311), VCon{F32.neg(0.085486), VCon{F32.neg(0.264778), VCon{F32.neg(0.279163), VCon{F32.neg(0.169500), VCon{F32.neg(0.085291), VCon{F32.neg(0.337160), VCon{F32.neg(0.256913), VCon{F32.neg(0.060409), VCon{F32.neg(0.059992), VCon{F32.neg(0.406600), VCon{F32.neg(0.058665), VCon{0.046077, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.262367, VCon{0.279756, VCon{0.333030, VCon{0.194536, VCon{F32.neg(0.107457), VCon{F32.neg(0.057879), VCon{F32.neg(0.074311), VCon{F32.neg(0.179322), VCon{F32.neg(0.002174), VCon{0.219897, VCon{F32.neg(0.096024), VCon{0.143630, VCon{F32.neg(0.220089), VCon{0.283854, VCon{F32.neg(0.112885), VCon{0.325621, VCon{0.356295, VCon{0.006339, VCon{F32.neg(0.074614), VCon{F32.neg(0.000684), VCon{F32.neg(0.253545), VCon{0.004959, VCon{F32.neg(0.197341), VCon{F32.neg(0.550272), VCon{0.228541, VCon{F32.neg(0.186696), VCon{F32.neg(0.274078), VCon{F32.neg(0.142975), VCon{F32.neg(0.142991), VCon{0.455387, VCon{F32.neg(0.108302), VCon{F32.neg(0.029734), VCon{0.014908, VCon{0.002736, VCon{0.129169, VCon{F32.neg(0.158704), VCon{0.234651, VCon{0.166805, VCon{0.183857, VCon{F32.neg(0.047008), VCon{F32.neg(0.066494), VCon{0.345965, VCon{F32.neg(0.073488), VCon{0.016130, VCon{0.362212, VCon{F32.neg(0.318938), VCon{0.233414, VCon{F32.neg(0.073787), VCon{F32.neg(0.299400), VCon{0.200565, VCon{0.128624, VCon{F32.neg(0.021342), VCon{0.103379, VCon{0.122837, VCon{F32.neg(0.123662), VCon{F32.neg(0.024090), VCon{F32.neg(0.085218), VCon{F32.neg(0.389928), VCon{F32.neg(0.003010), VCon{0.018816, VCon{F32.neg(0.023606), VCon{0.122369, VCon{F32.neg(0.178835), VCon{F32.neg(0.027997), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.258527), VCon{F32.neg(0.255269), VCon{F32.neg(0.107066), VCon{F32.neg(0.254687), VCon{F32.neg(0.056850), VCon{0.061397, VCon{0.049837, VCon{0.038239, VCon{F32.neg(0.259745), VCon{0.016450, VCon{F32.neg(0.204095), VCon{F32.neg(0.178309), VCon{F32.neg(0.102884), VCon{F32.neg(0.147946), VCon{F32.neg(0.047674), VCon{F32.neg(0.143287), VCon{0.652681, VCon{0.087375, VCon{0.008174, VCon{0.284634, VCon{0.079825, VCon{F32.neg(0.097893), VCon{0.038303, VCon{0.026984, VCon{F32.neg(0.206744), VCon{0.162255, VCon{F32.neg(0.097382), VCon{0.038818, VCon{0.186812, VCon{F32.neg(0.314043), VCon{F32.neg(0.072221), VCon{0.065346, VCon{F32.neg(0.157155), VCon{F32.neg(0.012315), VCon{0.037562, VCon{F32.neg(0.113872), VCon{F32.neg(0.426073), VCon{F32.neg(0.196299), VCon{0.290656, VCon{0.107429, VCon{0.066695, VCon{0.109773, VCon{0.045276, VCon{0.189602, VCon{F32.neg(0.116475), VCon{F32.neg(0.170224), VCon{0.070390, VCon{0.115801, VCon{F32.neg(0.038411), VCon{F32.neg(0.179093), VCon{F32.neg(0.557668), VCon{0.412951, VCon{0.079487, VCon{F32.neg(0.227963), VCon{F32.neg(0.215594), VCon{0.150355, VCon{F32.neg(0.186250), VCon{F32.neg(0.099113), VCon{F32.neg(0.049217), VCon{F32.neg(0.088279), VCon{0.389870, VCon{0.148215, VCon{0.052118, VCon{0.095000, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.145076), VCon{F32.neg(0.085416), VCon{0.043408, VCon{F32.neg(0.008271), VCon{F32.neg(0.052321), VCon{F32.neg(0.219573), VCon{0.156028, VCon{0.062438, VCon{0.109272, VCon{0.201871, VCon{F32.neg(0.050786), VCon{F32.neg(0.079549), VCon{0.003690, VCon{0.267860, VCon{F32.neg(0.077043), VCon{F32.neg(0.349980), VCon{0.146720, VCon{0.011385, VCon{0.060838, VCon{0.029844, VCon{F32.neg(0.182697), VCon{F32.neg(0.123230), VCon{0.414775, VCon{F32.neg(0.011280), VCon{F32.neg(0.276809), VCon{F32.neg(0.054614), VCon{0.292722, VCon{0.028361, VCon{0.199473, VCon{F32.neg(0.333620), VCon{F32.neg(0.087614), VCon{F32.neg(0.075455), VCon{0.084538, VCon{F32.neg(0.211459), VCon{0.030246, VCon{0.171660, VCon{F32.neg(0.146858), VCon{F32.neg(0.220450), VCon{0.144730, VCon{F32.neg(0.122231), VCon{F32.neg(0.067424), VCon{F32.neg(0.025665), VCon{F32.neg(0.168715), VCon{0.178761, VCon{F32.neg(0.124139), VCon{F32.neg(0.214857), VCon{0.051367, VCon{F32.neg(0.036558), VCon{0.269813, VCon{F32.neg(0.035842), VCon{F32.neg(0.198077), VCon{0.208865, VCon{0.032443, VCon{0.000892, VCon{F32.neg(0.092514), VCon{0.173370, VCon{0.085766, VCon{0.071488, VCon{0.045063, VCon{F32.neg(0.009022), VCon{F32.neg(0.116910), VCon{F32.neg(0.096990), VCon{F32.neg(0.185140), VCon{F32.neg(0.041498), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.124735), VCon{F32.neg(0.320120), VCon{F32.neg(0.446079), VCon{F32.neg(0.115129), VCon{F32.neg(0.041503), VCon{F32.neg(0.039997), VCon{0.540423, VCon{0.224316, VCon{F32.neg(0.034723), VCon{F32.neg(0.010548), VCon{F32.neg(0.133087), VCon{F32.neg(0.183981), VCon{F32.neg(0.249815), VCon{0.379504, VCon{F32.neg(0.218977), VCon{F32.neg(0.079699), VCon{0.006757, VCon{0.186644, VCon{0.103532, VCon{0.135419, VCon{F32.neg(0.215167), VCon{F32.neg(0.222126), VCon{0.296746, VCon{0.246282, VCon{F32.neg(0.243325), VCon{0.005717, VCon{0.058927, VCon{0.221221, VCon{0.517798, VCon{0.082553, VCon{0.157412, VCon{F32.neg(0.120826), VCon{F32.neg(0.065177), VCon{F32.neg(0.303300), VCon{0.015825, VCon{F32.neg(0.065232), VCon{0.017972, VCon{F32.neg(0.308854), VCon{F32.neg(0.227350), VCon{0.083176, VCon{F32.neg(0.049584), VCon{F32.neg(0.472442), VCon{0.233598, VCon{F32.neg(0.423001), VCon{F32.neg(0.100198), VCon{F32.neg(0.055575), VCon{F32.neg(0.256626), VCon{0.015520, VCon{0.285587, VCon{F32.neg(0.351421), VCon{0.166086, VCon{F32.neg(0.173968), VCon{0.090887, VCon{F32.neg(0.384879), VCon{F32.neg(0.463262), VCon{0.083022, VCon{F32.neg(0.112124), VCon{0.391123, VCon{F32.neg(0.055768), VCon{F32.neg(0.069994), VCon{F32.neg(0.093911), VCon{F32.neg(0.122532), VCon{F32.neg(0.114702), VCon{0.236277, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.035008), VCon{F32.neg(0.053783), VCon{F32.neg(0.363225), VCon{F32.neg(0.011078), VCon{F32.neg(0.198371), VCon{0.212410, VCon{F32.neg(0.389776), VCon{F32.neg(0.136438), VCon{0.006269, VCon{0.024258, VCon{F32.neg(0.053145), VCon{0.032368, VCon{F32.neg(0.034593), VCon{F32.neg(0.145807), VCon{F32.neg(0.038314), VCon{0.007724, VCon{0.284390, VCon{0.128576, VCon{F32.neg(0.170851), VCon{0.401727, VCon{F32.neg(0.107834), VCon{F32.neg(0.140994), VCon{0.095842, VCon{0.067207, VCon{0.060330, VCon{0.037130, VCon{F32.neg(0.156493), VCon{0.032798, VCon{F32.neg(0.009389), VCon{F32.neg(0.164509), VCon{F32.neg(0.013122), VCon{0.159065, VCon{F32.neg(0.035781), VCon{0.035702, VCon{0.171762, VCon{F32.neg(0.101578), VCon{F32.neg(0.337844), VCon{0.074918, VCon{F32.neg(0.284633), VCon{F32.neg(0.013258), VCon{0.266624, VCon{0.032539, VCon{F32.neg(0.012476), VCon{0.372339, VCon{0.055826, VCon{F32.neg(0.112683), VCon{F32.neg(0.107830), VCon{0.304658, VCon{F32.neg(0.140995), VCon{F32.neg(0.055245), VCon{F32.neg(0.387806), VCon{0.402900, VCon{0.088765, VCon{0.022666, VCon{F32.neg(0.164361), VCon{0.285646, VCon{0.067553, VCon{F32.neg(0.109529), VCon{F32.neg(0.249630), VCon{0.085371, VCon{0.379830, VCon{0.490762, VCon{F32.neg(0.065013), VCon{0.003557, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_b_mlp_proj() -> Vec:
  VCon{F32.neg(0.196608), VCon{0.004328, VCon{0.086428, VCon{F32.neg(0.127205), VCon{0.016077, VCon{F32.neg(0.079775), VCon{0.216116, VCon{F32.neg(0.063847), VCon{0.049640, VCon{0.152077, VCon{0.172733, VCon{0.186321, VCon{F32.neg(0.132900), VCon{F32.neg(0.207186), VCon{F32.neg(0.227219), VCon{0.129399, VNil{}}}}}}}}}}}}}}}}}

def param_ln_f_g() -> Vec:
  VCon{2.023376, VCon{1.947960, VCon{1.739676, VCon{1.940124, VCon{1.848505, VCon{1.882467, VCon{1.788974, VCon{1.838068, VCon{1.887949, VCon{1.884534, VCon{1.695491, VCon{1.936389, VCon{1.917717, VCon{1.739216, VCon{1.828265, VCon{1.839319, VNil{}}}}}}}}}}}}}}}}}

def param_ln_f_b() -> Vec:
  VCon{F32.neg(0.764979), VCon{F32.neg(0.125397), VCon{0.450419, VCon{F32.neg(0.304224), VCon{0.064571, VCon{F32.neg(0.453748), VCon{0.597436, VCon{0.230475, VCon{F32.neg(0.062639), VCon{0.648602, VCon{0.597771, VCon{0.665760, VCon{F32.neg(0.588853), VCon{F32.neg(0.650279), VCon{F32.neg(0.577240), VCon{0.559374, VNil{}}}}}}}}}}}}}}}}}

def param_lm_head() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.230509, VCon{F32.neg(0.569929), VCon{0.631178, VCon{F32.neg(0.538692), VCon{F32.neg(0.194463), VCon{0.102079, VCon{0.618830, VCon{0.579975, VCon{F32.neg(0.587564), VCon{0.483350, VCon{0.421099, VCon{0.341520, VCon{F32.neg(0.551534), VCon{F32.neg(0.232892), VCon{F32.neg(0.494657), VCon{0.286770, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.729684), VCon{0.362953, VCon{0.118031, VCon{0.328889, VCon{F32.neg(0.369403), VCon{F32.neg(0.609305), VCon{F32.neg(0.558186), VCon{0.015950, VCon{0.577762, VCon{0.866090, VCon{0.243268, VCon{0.256674, VCon{F32.neg(0.781009), VCon{F32.neg(0.036647), VCon{0.594819, VCon{F32.neg(0.596416), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.518690, VCon{0.158286, VCon{0.047624, VCon{0.344778, VCon{0.154769, VCon{0.079137, VCon{F32.neg(0.126148), VCon{F32.neg(0.239850), VCon{0.099910, VCon{F32.neg(0.258388), VCon{F32.neg(0.473019), VCon{F32.neg(0.386008), VCon{0.140121, VCon{F32.neg(0.074918), VCon{0.359858, VCon{F32.neg(0.358449), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.501833, VCon{F32.neg(0.060193), VCon{F32.neg(0.263806), VCon{0.379527, VCon{0.023275, VCon{0.306733, VCon{F32.neg(0.431119), VCon{F32.neg(0.256226), VCon{0.119552, VCon{F32.neg(0.504685), VCon{F32.neg(0.428595), VCon{F32.neg(0.490704), VCon{0.489143, VCon{0.453012, VCon{0.420321, VCon{F32.neg(0.162308), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.742825), VCon{F32.neg(0.518810), VCon{F32.neg(0.626362), VCon{F32.neg(0.429438), VCon{0.765808, VCon{F32.neg(0.429393), VCon{F32.neg(0.156085), VCon{0.403783, VCon{F32.neg(0.296399), VCon{F32.neg(0.142205), VCon{0.187261, VCon{F32.neg(0.148112), VCon{F32.neg(0.281072), VCon{0.641167, VCon{F32.neg(0.240958), VCon{0.617954, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.567932, VCon{0.643967, VCon{F32.neg(0.411150), VCon{0.409205, VCon{0.570523, VCon{F32.neg(0.128715), VCon{0.471955, VCon{F32.neg(0.614011), VCon{0.527731, VCon{0.239032, VCon{F32.neg(0.463628), VCon{F32.neg(0.420642), VCon{0.427739, VCon{F32.neg(0.639298), VCon{F32.neg(0.593195), VCon{0.181368, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{1.073671, VCon{0.013477, VCon{0.698626, VCon{0.656006, VCon{F32.neg(0.721525), VCon{F32.neg(0.753895), VCon{F32.neg(0.199103), VCon{0.302323, VCon{F32.neg(0.287389), VCon{F32.neg(0.176547), VCon{0.047733, VCon{0.271530, VCon{F32.neg(0.306153), VCon{0.205975, VCon{0.314445, VCon{F32.neg(0.786332), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.725063, VCon{F32.neg(0.065550), VCon{0.531539, VCon{F32.neg(0.138274), VCon{F32.neg(0.664977), VCon{F32.neg(0.124408), VCon{0.348485, VCon{0.411294, VCon{F32.neg(0.255674), VCon{0.560118, VCon{0.165414, VCon{F32.neg(1.080981), VCon{F32.neg(0.607011), VCon{0.525973, VCon{0.685898, VCon{F32.neg(0.678386), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.497042, VCon{F32.neg(0.043706), VCon{F32.neg(0.269620), VCon{0.375498, VCon{0.031643, VCon{0.256754, VCon{F32.neg(0.449313), VCon{F32.neg(0.236941), VCon{0.110764, VCon{F32.neg(0.506101), VCon{F32.neg(0.368964), VCon{F32.neg(0.431462), VCon{0.448110, VCon{0.459664, VCon{0.397646, VCon{F32.neg(0.147639), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.603449), VCon{0.347488, VCon{F32.neg(0.070220), VCon{0.930950, VCon{F32.neg(0.893707), VCon{F32.neg(0.117228), VCon{0.179683, VCon{F32.neg(0.197897), VCon{F32.neg(0.086651), VCon{F32.neg(0.091483), VCon{0.020928, VCon{0.757507, VCon{F32.neg(0.281873), VCon{F32.neg(0.042562), VCon{0.606788, VCon{0.043416, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.552170, VCon{F32.neg(0.054983), VCon{F32.neg(0.290567), VCon{0.352616, VCon{0.021865, VCon{0.263544, VCon{F32.neg(0.506906), VCon{F32.neg(0.224140), VCon{0.102821, VCon{F32.neg(0.521345), VCon{F32.neg(0.382453), VCon{F32.neg(0.460099), VCon{0.483263, VCon{0.433532, VCon{0.428721, VCon{F32.neg(0.165053), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.516326, VCon{F32.neg(0.020280), VCon{F32.neg(0.213495), VCon{0.377535, VCon{0.060237, VCon{0.308373, VCon{F32.neg(0.450245), VCon{F32.neg(0.227185), VCon{0.135512, VCon{F32.neg(0.524363), VCon{F32.neg(0.366344), VCon{F32.neg(0.445318), VCon{0.507188, VCon{0.480948, VCon{0.446872, VCon{F32.neg(0.138017), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.601026), VCon{0.421364, VCon{0.430481, VCon{F32.neg(0.624794), VCon{0.701813, VCon{F32.neg(0.197018), VCon{F32.neg(0.098223), VCon{F32.neg(0.671725), VCon{0.690034, VCon{0.461724, VCon{F32.neg(0.097608), VCon{F32.neg(0.656532), VCon{0.484346, VCon{F32.neg(0.521218), VCon{0.334880, VCon{F32.neg(0.368016), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.547978, VCon{F32.neg(0.028270), VCon{F32.neg(0.252550), VCon{0.388137, VCon{0.064700, VCon{0.277691, VCon{F32.neg(0.478667), VCon{F32.neg(0.245422), VCon{0.111122, VCon{F32.neg(0.503030), VCon{F32.neg(0.386521), VCon{F32.neg(0.450968), VCon{0.488073, VCon{0.445687, VCon{0.460691, VCon{F32.neg(0.149234), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.369076), VCon{0.571090, VCon{F32.neg(0.497300), VCon{F32.neg(0.271804), VCon{0.697514, VCon{F32.neg(0.209390), VCon{0.556635, VCon{0.119155, VCon{F32.neg(0.449297), VCon{F32.neg(0.092858), VCon{0.209899, VCon{0.816198, VCon{0.446426, VCon{F32.neg(0.274525), VCon{F32.neg(0.767313), VCon{0.688667, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.423245), VCon{0.049423, VCon{F32.neg(0.715018), VCon{0.352782, VCon{F32.neg(0.419563), VCon{0.779211, VCon{0.171598, VCon{0.487545, VCon{F32.neg(0.641261), VCon{F32.neg(0.350589), VCon{F32.neg(0.077344), VCon{F32.neg(0.156899), VCon{0.181278, VCon{0.542235, VCon{0.289853, VCon{F32.neg(0.340718), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.536798), VCon{0.698291, VCon{F32.neg(0.284369), VCon{F32.neg(0.020120), VCon{F32.neg(0.820187), VCon{F32.neg(0.079134), VCon{0.386410, VCon{0.450107, VCon{F32.neg(0.487505), VCon{0.614165, VCon{F32.neg(0.219134), VCon{F32.neg(0.528417), VCon{F32.neg(0.191784), VCon{0.586749, VCon{0.280185, VCon{F32.neg(0.088529), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.540734, VCon{F32.neg(0.030822), VCon{F32.neg(0.239883), VCon{0.358045, VCon{0.029604, VCon{0.296500, VCon{F32.neg(0.455411), VCon{F32.neg(0.213748), VCon{0.117175, VCon{F32.neg(0.490120), VCon{F32.neg(0.402010), VCon{F32.neg(0.473502), VCon{0.467376, VCon{0.464054, VCon{0.412708, VCon{F32.neg(0.167284), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.790142), VCon{0.076059, VCon{0.117062, VCon{0.132567, VCon{F32.neg(0.384863), VCon{0.099529, VCon{F32.neg(0.576952), VCon{0.692289, VCon{F32.neg(1.014255), VCon{F32.neg(0.238274), VCon{F32.neg(0.053503), VCon{F32.neg(0.197801), VCon{F32.neg(0.143191), VCon{0.481917, VCon{0.725843, VCon{F32.neg(0.005440), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.459331), VCon{F32.neg(0.646389), VCon{F32.neg(0.101622), VCon{0.509094, VCon{F32.neg(0.391907), VCon{F32.neg(0.465286), VCon{F32.neg(0.174023), VCon{F32.neg(0.476241), VCon{0.723181, VCon{F32.neg(0.049735), VCon{0.733048, VCon{0.803927, VCon{0.050046, VCon{F32.neg(0.085747), VCon{F32.neg(0.684722), VCon{F32.neg(0.191669), VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.796539, VCon{F32.neg(0.234333), VCon{0.282990, VCon{0.458074, VCon{F32.neg(0.715959), VCon{0.031031, VCon{0.047602, VCon{F32.neg(0.646187), VCon{0.299143, VCon{F32.neg(0.559071), VCon{F32.neg(0.378227), VCon{0.417964, VCon{0.791175, VCon{F32.neg(0.293843), VCon{F32.neg(0.599317), VCon{0.346883, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.927184), VCon{0.677647, VCon{F32.neg(0.704477), VCon{F32.neg(0.293617), VCon{0.000218, VCon{0.400674, VCon{F32.neg(0.883944), VCon{F32.neg(0.038012), VCon{0.813261, VCon{0.169244, VCon{0.598483, VCon{0.701971, VCon{0.602403, VCon{F32.neg(0.094147), VCon{F32.neg(0.014421), VCon{0.004071, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.535587, VCon{F32.neg(0.007113), VCon{F32.neg(0.260145), VCon{0.361026, VCon{0.080762, VCon{0.305594, VCon{F32.neg(0.450278), VCon{F32.neg(0.281598), VCon{0.133598, VCon{F32.neg(0.506555), VCon{F32.neg(0.416464), VCon{F32.neg(0.429768), VCon{0.452217, VCon{0.464629, VCon{0.494317, VCon{F32.neg(0.165315), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.543394, VCon{F32.neg(0.036285), VCon{F32.neg(0.276159), VCon{0.346318, VCon{0.044942, VCon{0.278096, VCon{F32.neg(0.480584), VCon{F32.neg(0.241136), VCon{0.125396, VCon{F32.neg(0.524972), VCon{F32.neg(0.373807), VCon{F32.neg(0.449914), VCon{0.496610, VCon{0.468166, VCon{0.458060, VCon{F32.neg(0.155896), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.553786, VCon{F32.neg(0.029609), VCon{F32.neg(0.255964), VCon{0.370847, VCon{0.034644, VCon{0.291564, VCon{F32.neg(0.417286), VCon{F32.neg(0.230076), VCon{0.142495, VCon{F32.neg(0.513863), VCon{F32.neg(0.407865), VCon{F32.neg(0.482884), VCon{0.479628, VCon{0.476808, VCon{0.458331, VCon{F32.neg(0.130969), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.489498, VCon{F32.neg(0.051193), VCon{F32.neg(0.207873), VCon{0.388775, VCon{0.030949, VCon{0.291610, VCon{F32.neg(0.484318), VCon{F32.neg(0.189062), VCon{0.122904, VCon{F32.neg(0.511469), VCon{F32.neg(0.404553), VCon{F32.neg(0.467689), VCon{0.462194, VCon{0.453673, VCon{0.370508, VCon{F32.neg(0.157526), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.561409, VCon{F32.neg(0.024647), VCon{F32.neg(0.270024), VCon{0.364029, VCon{0.040622, VCon{0.262411, VCon{F32.neg(0.446134), VCon{F32.neg(0.254624), VCon{0.094605, VCon{F32.neg(0.507751), VCon{F32.neg(0.350535), VCon{F32.neg(0.474135), VCon{0.462553, VCon{0.489602, VCon{0.414475, VCon{F32.neg(0.178540), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.287432, VCon{F32.neg(0.712284), VCon{0.589174, VCon{F32.neg(0.868244), VCon{F32.neg(0.157593), VCon{0.710952, VCon{0.072358, VCon{F32.neg(0.610547), VCon{0.307358, VCon{F32.neg(0.337212), VCon{0.309292, VCon{0.069010, VCon{0.351116, VCon{F32.neg(0.460329), VCon{F32.neg(0.320431), VCon{0.886184, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.499293, VCon{F32.neg(0.060011), VCon{F32.neg(0.229552), VCon{0.394007, VCon{0.042161, VCon{0.315268, VCon{F32.neg(0.496220), VCon{F32.neg(0.231090), VCon{0.123797, VCon{F32.neg(0.530294), VCon{F32.neg(0.416513), VCon{F32.neg(0.486513), VCon{0.481613, VCon{0.440275, VCon{0.421603, VCon{F32.neg(0.157666), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.526003, VCon{F32.neg(0.018966), VCon{F32.neg(0.209498), VCon{0.371186, VCon{0.066981, VCon{0.299303, VCon{F32.neg(0.497291), VCon{F32.neg(0.211515), VCon{0.129340, VCon{F32.neg(0.535265), VCon{F32.neg(0.362737), VCon{F32.neg(0.464975), VCon{0.452132, VCon{0.474007, VCon{0.427060, VCon{F32.neg(0.157591), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.524002, VCon{F32.neg(0.034048), VCon{F32.neg(0.224669), VCon{0.386050, VCon{0.059118, VCon{0.294364, VCon{F32.neg(0.473671), VCon{F32.neg(0.236645), VCon{0.111179, VCon{F32.neg(0.495624), VCon{F32.neg(0.384326), VCon{F32.neg(0.472766), VCon{0.505582, VCon{0.465952, VCon{0.428683, VCon{F32.neg(0.163115), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.538797, VCon{F32.neg(0.081501), VCon{F32.neg(0.277410), VCon{0.374503, VCon{0.012380, VCon{0.289151, VCon{F32.neg(0.483823), VCon{F32.neg(0.187796), VCon{0.120549, VCon{F32.neg(0.513066), VCon{F32.neg(0.388432), VCon{F32.neg(0.479629), VCon{0.486066, VCon{0.435490, VCon{0.420002, VCon{F32.neg(0.147184), VNil{}}}}}}}}}}}}}}}}}}
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
      k = linear_layer(normed, param_wk(), param_bk())
      v = linear_layer(normed, param_wv(), param_bv())
      KVCon{KVP{k, v}, compute_kv_list(rest, (pos + 1 : U32))}

# Compute attention score for each past token against query Q
def compute_scores(+q: Vec, kvs: KVList) -> Vec:
  match kvs:
    case KVNil{}:
      VNil{}
    case KVCon{KVP{+k, _}, rest}:
      scale : F32 = 0.250000
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
  +q : Vec = linear_layer(normed1, param_wq(), param_bq())
  
  # 4. Multi-token Causal Self-Attention over past tokens
  +kvs : KVList = compute_kv_list(tokens, 0)
  scores : Vec = compute_scores(q, kvs)
  +attn_weights : Vec = vec_softmax(scores)
  context : Vec = apply_attention(attn_weights, kvs)
  attn_out : Vec = linear_layer(context, param_w_attn_proj(), param_b_attn_proj())
  
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

# Autoregressive generation loop in Bend (Structural Induction on Nat):
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
    
    IO.print("--- Karpathy nanoGPT Text Generation in Bend ---")
    IO.print("Prompt:   " ++ print_tokens(prompt))
    
    # Autoregressively generate next 9 tokens using GPU tree operations:
    gen : Tokens = generate_tokens(9n, prompt)
    
    IO.print("Output:   " ++ print_tokens(gen))
    IO.print("------------------------------------------------")
