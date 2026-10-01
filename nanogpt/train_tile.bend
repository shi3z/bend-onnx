# ==============================================================================
# nanoGPT TRAINING in Bend, v2: 16-wide TILE data layout
#
# Same model, same hand-written backward, same AdamW as train_lib.bend (v1), but every
# vector is a list of T16 nodes (16 contiguous F32 fields) instead of a cons list of
# scalars. See nanogpt/tile_gen.py for why: one load = 16 values, dot16 = straight-line
# code, 16x fewer heap cells, and matmul output is produced 16 columns at a time.
#
# Assembled by nanogpt/bench_train.py:
#   size constants, tile primitives (tile_gen.tile_ops), THIS FILE up to the first marker,
#   tile matmul kernels (tile_gen.tile_mat_ops), the rest, generated data, main.
# ==============================================================================

# --- flat scalar vectors (parameters / Adam state live in plain F32 lists) -------------
# A GPU lane's continuation stack is ~2k frames: long lists are walked by tail-recursive
# accumulate-and-reverse loops, never by `VCon{h, f(t)}`.
def vrev_onto(x: Vec, acc: Vec) -> Vec:
  match x:
    case VNil{}:
      acc
    case VCon{h, t}:
      vrev_onto(t, VCon{h, acc})

def vrev(x: Vec) -> Vec:
  vrev_onto(x, VNil{})

def vcat(x: Vec, y: Vec) -> Vec:
  vrev_onto(vrev_onto(x, VNil{}), y)

def vec_zeros_acc(+n: Nat, acc: Vec) -> Vec:
  match n:
    case 0n:
      acc
    case 1n+p:
      vec_zeros_acc(p, VCon{0.0, acc})

def vec_zeros(+n: Nat) -> Vec:
  vec_zeros_acc(n, VNil{})

def vflat_scale_acc(+s: F32, v: Vec, acc: Vec) -> Vec:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vflat_scale_acc(s, t, VCon{(s * h : F32), acc})

def vflat_scale(+s: F32, v: Vec) -> Vec:
  vrev(vflat_scale_acc(s, v, VNil{}))

def vtake_acc(+n: Nat, +v: Vec, acc: Vec) -> Vec:
  match n v:
    case 1n+p VCon{h, t}:
      vtake_acc(p, t, VCon{h, acc})
    case _ _:
      acc

def vtake(+n: Nat, +v: Vec) -> Vec:
  vrev(vtake_acc(n, v, VNil{}))

def vdrop(+n: Nat, v: Vec) -> Vec:
  match n v:
    case 1n+p VCon{_, t}:
      vdrop(p, t)
    case _ _:
      v

def vec_get(+v: Vec, +idx: U32) -> F32:
  match v idx:
    case VCon{+h, _} 0:
      h
    case VCon{_, t} _:
      vec_get(t, U32.sub(idx, 1))
    case _ _:
      0.0

# --- TV: vectors as lists of tiles ---------------------------------------------------------
def tv_add(x: TV, y: TV) -> TV:
  match x y:
    case QCon{xh, xt} QCon{yh, yt}:
      QCon{t_add(xh, yh), tv_add(xt, yt)}
    case _ _:
      QNil{}

def tv_sub(x: TV, y: TV) -> TV:
  match x y:
    case QCon{xh, xt} QCon{yh, yt}:
      QCon{t_sub(xh, yh), tv_sub(xt, yt)}
    case _ _:
      QNil{}

def tv_mul(x: TV, y: TV) -> TV:
  match x y:
    case QCon{xh, xt} QCon{yh, yt}:
      QCon{t_mul(xh, yh), tv_mul(xt, yt)}
    case _ _:
      QNil{}

def tv_scale(+s: F32, x: TV) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_scale(s, h), tv_scale(s, t)}

def tv_addc(+s: F32, x: TV) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_addc(s, h), tv_addc(s, t)}

def tv_gelu(x: TV) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_gelu(h), tv_gelu(t)}

def tv_gelu_bwd(f: TV, dh: TV) -> TV:
  match f dh:
    case QCon{fh, ft} QCon{dh_, dt}:
      QCon{t_gelu_bwd(fh, dh_), tv_gelu_bwd(ft, dt)}
    case _ _:
      QNil{}

def tv_expm(+m: F32, x: TV) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_expm(m, h), tv_expm(m, t)}

def tv_mask(x: TV, +lim: U32, +base: U32) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_mask(lim, base, h), tv_mask(t, lim, (base + 16 : U32))}

def tv_sum(x: TV, +acc: F32) -> F32:
  match x:
    case QNil{}:
      acc
    case QCon{h, t}:
      tv_sum(t, (acc + t_sum(h) : F32))

def tv_dot(x: TV, y: TV, +acc: F32) -> F32:
  match x y:
    case QCon{xh, xt} QCon{yh, yt}:
      tv_dot(xt, yt, (acc + t_dot(xh, yh) : F32))
    case _ _:
      acc

def tv_max_acc(x: TV, +acc: F32) -> F32:
  match x:
    case QNil{}:
      acc
    case QCon{h, t}:
      tv_max_acc(t, F32.max(acc, t_max(h)))

def tv_max(x: TV) -> F32:
  match x:
    case QNil{}:
      0.0
    case QCon{h, t}:
      tv_max_acc(t, t_max(h))

def tv_zeros(+w: Nat) -> TV:
  match w:
    case 0n:
      QNil{}
    case 1n+p:
      QCon{t_zero(), tv_zeros(p)}

def tv_onehot(+w: Nat, +y: U32) -> TV:
  match w:
    case 0n:
      QNil{}
    case 1n+p:
      QCon{t_onehot(y), tv_onehot(p, U32.sub(y, 16))}

def tv_cat(x: TV, y: TV) -> TV:
  match x:
    case QNil{}:
      y
    case QCon{h, t}:
      QCon{h, tv_cat(t, y)}

def tv_to_rows(x: TV) -> Mat:
  match x:
    case QNil{}:
      MNil{}
    case QCon{h, t}:
      MCon{QCon{h, QNil{}}, tv_to_rows(t)}

def tv_of(+w: Nat, +v: Vec) -> TV:
  match w:
    case 0n:
      QNil{}
    case 1n+p:
      QCon{t16_of(v), tv_of(p, vdrop(16n, v))}

def tv_flat_onto(x: TV, tail: Vec) -> Vec:
  match x:
    case QNil{}:
      tail
    case QCon{h, t}:
      t16_onto(h, tv_flat_onto(t, tail))

def tv_adam_m(+s: F32, +b1: F32, +omb: F32, g: TV, m: TV) -> TV:
  match g m:
    case QCon{gh, gt} QCon{mh, mt}:
      QCon{t_adam_m(s, b1, omb, gh, mh), tv_adam_m(s, b1, omb, gt, mt)}
    case _ _:
      QNil{}

def tv_adam_v(+s: F32, +b2: F32, +omb: F32, g: TV, v: TV) -> TV:
  match g v:
    case QCon{gh, gt} QCon{vh, vt}:
      QCon{t_adam_v(s, b2, omb, gh, vh), tv_adam_v(s, b2, omb, gt, vt)}
    case _ _:
      QNil{}

def tv_adam_p(+lr: F32, +c1: F32, +c2: F32, +eps: F32, p: TV, m: TV, v: TV) -> TV:
  match p m v:
    case QCon{ph, pt} QCon{mh, mt} QCon{vh, vt}:
      QCon{t_adam_p(lr, c1, c2, eps, ph, mh, vh), tv_adam_p(lr, c1, c2, eps, pt, mt, vt)}
    case _ _ _:
      QNil{}

# --- Mat: lists of TV rows ---------------------------------------------------------------------
def mat_zeros(+r: Nat, +w: Nat) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MCon{tv_zeros(w), mat_zeros(p, w)}

# r rows, w tiles per row, rl = 16*w scalars per row
def mat_of(+r: Nat, +w: Nat, +rl: Nat, +v: Vec) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MCon{tv_of(w, v), mat_of(p, w, rl, vdrop(rl, v))}

def mat_adam_m(+s: F32, +b1: F32, +omb: F32, g: Mat, m: Mat) -> Mat:
  match g m:
    case MCon{gr, gt} MCon{mr, mt}:
      MCon{tv_adam_m(s, b1, omb, gr, mr), mat_adam_m(s, b1, omb, gt, mt)}
    case _ _:
      MNil{}

def mat_adam_v(+s: F32, +b2: F32, +omb: F32, g: Mat, v: Mat) -> Mat:
  match g v:
    case MCon{gr, gt} MCon{vr, vt}:
      MCon{tv_adam_v(s, b2, omb, gr, vr), mat_adam_v(s, b2, omb, gt, vt)}
    case _ _:
      MNil{}

def mat_adam_p(+lr: F32, +c1: F32, +c2: F32, +eps: F32, p: Mat, m: Mat, v: Mat) -> Mat:
  match p m v:
    case MCon{pr, pt} MCon{mr, mt} MCon{vr, vt}:
      MCon{tv_adam_p(lr, c1, c2, eps, pr, mr, vr), mat_adam_p(lr, c1, c2, eps, pt, mt, vt)}
    case _ _ _:
      MNil{}

def mat_flat_onto(m: Mat, tail: Vec) -> Vec:
  match m:
    case MNil{}:
      tail
    case MCon{r, rt}:
      tv_flat_onto(r, mat_flat_onto(rt, tail))

def mat_add(a: Mat, b: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      MCon{tv_add(ar, br), mat_add(at, bt)}
    case _ _:
      MNil{}

def mat_mul(a: Mat, b: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      MCon{tv_mul(ar, br), mat_mul(at, bt)}
    case _ _:
      MNil{}

def mat_scale(+s: F32, m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_scale(s, r), mat_scale(s, rt)}

def mat_add_bias(m: Mat, +b: TV) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_add(r, b), mat_add_bias(rt, b)}

def mat_colsum_acc(m: Mat, acc: TV) -> TV:
  match m:
    case MNil{}:
      acc
    case MCon{r, rt}:
      mat_colsum_acc(rt, tv_add(acc, r))

def mat_colsum(m: Mat, +w: Nat) -> TV:
  mat_colsum_acc(m, tv_zeros(w))

def mat_cat(a: Mat, b: Mat) -> Mat:
  match a:
    case MNil{}:
      b
    case MCon{r, rt}:
      MCon{r, mat_cat(rt, b)}

def mat_len_acc(m: Mat, +acc: U32) -> U32:
  match m:
    case MNil{}:
      acc
    case MCon{_, rt}:
      mat_len_acc(rt, (acc + 1 : U32))

def mat_take(m: Mat, +n: U32) -> Mat:
  match m n:
    case MCon{r, rt} 0:
      MNil{}
    case MCon{r, rt} _:
      MCon{r, mat_take(rt, U32.sub(n, 1))}
    case _ _:
      MNil{}

def mat_drop(m: Mat, +n: U32) -> Mat:
  match m n:
    case MCon{_, rt} 0:
      m
    case MCon{_, rt} _:
      mat_drop(rt, U32.sub(n, 1))
    case _ _:
      m

# row-wise concatenation of two matrices with the same number of rows (b may be empty)
def mat_hcat(a: Mat, b: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      MCon{tv_cat(ar, br), mat_hcat(at, bt)}
    case _ _:
      a

# pad with zero rows up to a multiple of 16 rows (w = tiles per row)
def mat_pad16(+m: Mat, +w: Nat) -> Mat:
  +n : U32 = mat_len_acc(m, 0)
  padn : U32 = U32.mod(U32.sub(16, U32.mod(n, 16)), 16)
  mat_cat(m, mat_zeros(U32.to_nat(padn), w))

# @@TILE_MAT_OPS@@

def mat_t(m: Mat, +w: Nat) -> Mat:
  mat_t_go(mat_pad16(m, w), w)

# kk = number of 16-wide tiles in the contraction dimension (selects the unrolled kernel)
def row_nt_k(+kk: U32, ar: TV, b: Mat) -> TV:
  match kk:
    case 1:
      row_nt1(ar, b)
    case 2:
      row_nt2(ar, b)
    case 4:
      row_nt4(ar, b)
    case _:
      row_nt(ar, b)

def mat_nt_seq(a: Mat, +b: Mat, +kk: U32) -> Mat:
  match a:
    case MNil{}:
      MNil{}
    case MCon{ar, at}:
      MCon{row_nt_k(kk, ar, b), mat_nt_seq(at, b, kk)}

# Row-parallel A B^T: split the rows of A in halves `d` times (2^d independent tasks).
def mat_nt_p(+d: Nat, +a: Mat, +b: Mat, +kk: U32) -> Mat:
  match d:
    case 0n:
      mat_nt_seq(a, b, kk)
    case 1n+k:
      +h : U32 = U32.div(mat_len_acc(a, 0), 2)
      l r = mat_nt_p(k, mat_take(a, h), b, kk) mat_nt_p(k, mat_drop(a, h), b, kk)
      mat_cat(l, r)

# A B^T. Rows of B must come in multiples of 16.
def mat_nt(a: Mat, +b: Mat, +kk: U32) -> Mat:
  mat_nt_p(fork_depth(), a, b, kk)

def mat_row_add(acc: Mat, +idx: U32, row: TV) -> Mat:
  match acc idx:
    case MCon{r, rt} 0:
      MCon{tv_add(r, row), rt}
    case MCon{r, rt} _:
      MCon{r, mat_row_add(rt, U32.sub(idx, 1), row)}
    case _ _:
      acc

# --- Deep copies (private parameters per fork-tree branch) -----------------------------------------
def tv_copy(x: TV) -> TV:
  match x:
    case QNil{}:
      QNil{}
    case QCon{h, t}:
      QCon{t_copy(h), tv_copy(t)}

def mat_copy(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_copy(r), mat_copy(rt)}

def mat_row_get(+m: Mat, +idx: U32) -> TV:
  match m idx:
    case MCon{+r, _} 0:
      r
    case MCon{_, rt} _:
      mat_row_get(rt, U32.sub(idx, 1))
    case _ _:
      QNil{}

# --- Embedding -------------------------------------------------------------------------------------
def embed(+wte: Mat, toks: Toks, wpe: Mat) -> Mat:
  match toks wpe:
    case TCon{+tk, tt} MCon{pr, pt}:
      MCon{tv_add(mat_row_get(wte, tk), pr), embed(wte, tt, pt)}
    case _ _:
      MNil{}

def scatter_add(toks: Toks, de: Mat, acc: Mat) -> Mat:
  match toks de:
    case TCon{+tk, tt} MCon{dr, dt}:
      scatter_add(tt, dt, mat_row_add(acc, tk, dr))
    case _ _:
      acc

# --- GELU ----------------------------------------------------------------------------------------------
def mat_gelu(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_gelu(r), mat_gelu(rt)}

def mat_gelu_bwd(f: Mat, dh: Mat) -> Mat:
  match f dh:
    case MCon{fr, ft} MCon{dr, dt}:
      MCon{tv_gelu_bwd(fr, dr), mat_gelu_bwd(ft, dt)}
    case _ _:
      MNil{}

# --- LayerNorm over rows (fwd cache = xhat, rstd) -----------------------------------------------------------
def row_rstd(+v: TV) -> F32:
  +m : F32 = (tv_sum(v, 0.0) * inv_c() : F32)
  +cen : TV = tv_addc(F32.neg(m), v)
  var : F32 = (tv_dot(cen, cen, 0.0) * inv_c() : F32)
  (1.0 / F32.sqrt((var + 0.00001 : F32)) : F32)

def row_xhat(+v: TV) -> TV:
  +m : F32 = (tv_sum(v, 0.0) * inv_c() : F32)
  r : F32 = row_rstd(v)
  tv_scale(r, tv_addc(F32.neg(m), v))

def mat_xhat(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{row_xhat(r), mat_xhat(rt)}

def mat_rstd(m: Mat) -> Vec:
  match m:
    case MNil{}:
      VNil{}
    case MCon{r, rt}:
      VCon{row_rstd(r), mat_rstd(rt)}

def ln_affine(xhat: Mat, +g: TV, +b: TV) -> Mat:
  match xhat:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_add(tv_mul(r, g), b), ln_affine(rt, g, b)}

# dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),  dxhat = dy * gamma
def row_ln_bwd(dy: TV, +xhat: TV, +rstd: F32, +g: TV) -> TV:
  +dxh : TV = tv_mul(dy, g)
  +m1 : F32 = (tv_sum(dxh, 0.0) * inv_c() : F32)
  +m2 : F32 = (tv_sum(tv_mul(dxh, xhat), 0.0) * inv_c() : F32)
  tv_scale(rstd, tv_sub(tv_addc(F32.neg(m1), dxh), tv_scale(m2, xhat)))

def mat_ln_bwd(dy: Mat, xhat: Mat, rstd: Vec, +g: TV) -> Mat:
  match dy xhat rstd:
    case MCon{d, dt} MCon{xh, xt} VCon{+r, rt}:
      MCon{row_ln_bwd(d, xh, r, g), mat_ln_bwd(dt, xt, rt, g)}
    case _ _ _:
      MNil{}

# --- Causal attention pieces ----------------------------------------------------------------------------------
def mat_mask(m: Mat, +t: U32) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_mask(r, t, 0), mat_mask(rt, (t + 1 : U32))}

def tv_softmax(+v: TV) -> TV:
  +m : F32 = tv_max(v)
  +e : TV = tv_expm(m, v)
  s : F32 = tv_sum(e, 0.0)
  tv_scale((1.0 / s : F32), e)

def mat_softmax(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{tv_softmax(r), mat_softmax(rt)}

# dS = A * (dA - rowsum(A * dA))
def row_softmax_bwd(+a: TV, +da: TV) -> TV:
  +d : F32 = tv_dot(a, da, 0.0)
  tv_mul(a, tv_addc(F32.neg(d), da))

def mat_softmax_bwd(a: Mat, da: Mat) -> Mat:
  match a da:
    case MCon{ar, at} MCon{dr, dt}:
      MCon{row_softmax_bwd(ar, dr), mat_softmax_bwd(at, dt)}
    case _ _:
      MNil{}

# --- Cross entropy ---------------------------------------------------------------------------------------------------
def row_ce(+p: TV, +y: U32) -> F32:
  pv : F32 = F32.max(tv_dot(p, tv_onehot(nVt(), y), 0.0), 0.000000000000000000000000000001)
  F32.neg(F32.log(pv))

def ce_loss(pr: Mat, tg: Toks, +acc: F32) -> F32:
  match pr tg:
    case MCon{pv, pt} TCon{+y, yt}:
      ce_loss(pt, yt, (acc + row_ce(pv, y) : F32))
    case _ _:
      acc

# d logits = softmax - onehot(y)
def mat_ce_grad(pr: Mat, tg: Toks) -> Mat:
  match pr tg:
    case MCon{pv, pt} TCon{+y, yt}:
      MCon{tv_sub(pv, tv_onehot(nVt(), y)), mat_ce_grad(pt, yt)}
    case _ _:
      MNil{}

# --- Parameters (flat Vec <-> record) and gradients ------------------------------------------------------------------------
# The *T fields are transposed copies of the weights, built once per step on the host.
type P is Data:
  P{wte: Mat, wpe: Mat, ln1g: TV, ln1b: TV,
    wq: Mat, bq: TV, wk: Mat, bk: TV, wv: Mat, bv: TV, wo: Mat, bo: TV,
    ln2g: TV, ln2b: TV, wfc: Mat, bfc: TV, wmp: Mat, bmp: TV,
    lnfg: TV, lnfb: TV,
    wteT: Mat, wqT: Mat, wkT: Mat, wvT: Mat, woT: Mat, wfcT: Mat, wmpT: Mat}

type G is Data:
  G{loss: F32, wte: Mat, wpe: Mat, ln1g: TV, ln1b: TV,
    wq: Mat, bq: TV, wk: Mat, bk: TV, wv: Mat, bv: TV, wo: Mat, bo: TV,
    ln2g: TV, ln2b: TV, wfc: Mat, bfc: TV, wmp: Mat, bmp: TV,
    lnfg: TV, lnfb: TV}

def lin_fwd(+x: Mat, +w: Mat, +b: TV, +kk: U32) -> Mat:
  mat_add_bias(mat_nt(x, w, kk), b)

# --- One sample: forward + backward -> G -------------------------------------------------------------------------------------
def sample_grad(+p: P, toks: Toks, tgts: Toks) -> G:
  match p:
    case P{+wte, +wpe, +ln1g, +ln1b, +wq, +bq, +wk, +bk, +wv, +bv, +wo, +bo, +ln2g, +ln2b, +wfc, +bfc, +wmp, +bmp, +lnfg, +lnfb, +wteT, +wqT, +wkT, +wvT, +woT, +wfcT, +wmpT}:
      +toks2 : Toks = toks
      # ---- forward ----
      +e : Mat = embed(wte, toks2, wpe)
      +xh1 : Mat = mat_xhat(e)
      +rs1 : Vec = mat_rstd(e)
      +n1 : Mat = ln_affine(xh1, ln1g, ln1b)
      +q : Mat = lin_fwd(n1, wq, bq, kC())
      +k : Mat = lin_fwd(n1, wk, bk, kC())
      +v : Mat = lin_fwd(n1, wv, bv, kC())
      +kp : Mat = mat_pad16(k, nCt())
      +vp : Mat = mat_pad16(v, nCt())
      +vT : Mat = mat_t(v, nCt())
      +a : Mat = mat_softmax(mat_mask(mat_scale(fscale(), mat_nt(q, kp, kC())), 0))
      +ctx : Mat = mat_nt(a, vT, 1)
      +o : Mat = lin_fwd(ctx, wo, bo, kC())
      +r1 : Mat = mat_add(e, o)
      +xh2 : Mat = mat_xhat(r1)
      +rs2 : Vec = mat_rstd(r1)
      +n2 : Mat = ln_affine(xh2, ln2g, ln2b)
      +f : Mat = lin_fwd(n2, wfc, bfc, kC())
      +h : Mat = mat_gelu(f)
      +mo : Mat = lin_fwd(h, wmp, bmp, kF())
      +r2 : Mat = mat_add(r1, mo)
      +xhf : Mat = mat_xhat(r2)
      +rsf : Vec = mat_rstd(r2)
      +nf : Mat = ln_affine(xhf, lnfg, lnfb)
      +logits : Mat = mat_nt(nf, wte, kC())
      +pr : Mat = mat_softmax(logits)
      +tg : Toks = tgts
      loss : F32 = ce_loss(pr, tg, 0.0)
      # ---- backward ----
      +dlog : Mat = mat_ce_grad(pr, tg)
      dwte_head : Mat = mat_nt(mat_t(dlog, nVt()), mat_t(nf, nCt()), 1)
      +dnf : Mat = mat_nt(dlog, wteT, kV())
      dlnfg : TV = mat_colsum(mat_mul(dnf, xhf), nCt())
      dlnfb : TV = mat_colsum(dnf, nCt())
      +dr2 : Mat = mat_ln_bwd(dnf, xhf, rsf, lnfg)
      dwmp : Mat = mat_nt(mat_t(dr2, nCt()), mat_t(h, nFt()), 1)
      dbmp : TV = mat_colsum(dr2, nCt())
      +dh : Mat = mat_nt(dr2, wmpT, kC())
      +df : Mat = mat_gelu_bwd(f, dh)
      dwfc : Mat = mat_nt(mat_t(df, nFt()), mat_t(n2, nCt()), 1)
      dbfc : TV = mat_colsum(df, nFt())
      +dn2 : Mat = mat_nt(df, wfcT, kF())
      dln2g : TV = mat_colsum(mat_mul(dn2, xh2), nCt())
      dln2b : TV = mat_colsum(dn2, nCt())
      +dr1 : Mat = mat_add(dr2, mat_ln_bwd(dn2, xh2, rs2, ln2g))
      dwo : Mat = mat_nt(mat_t(dr1, nCt()), mat_t(ctx, nCt()), 1)
      dbo : TV = mat_colsum(dr1, nCt())
      +dctx : Mat = mat_nt(dr1, woT, kC())
      +da : Mat = mat_nt(dctx, vp, kC())
      dv : Mat = mat_take(mat_nt(mat_t(a, 1n), mat_t(dctx, nCt()), 1), nT32())
      +ds : Mat = mat_softmax_bwd(a, da)
      dq : Mat = mat_scale(fscale(), mat_nt(ds, mat_t(k, nCt()), 1))
      dk : Mat = mat_take(mat_scale(fscale(), mat_nt(mat_t(ds, 1n), mat_t(q, nCt()), 1)), nT32())
      +dq2 : Mat = dq
      +dk2 : Mat = dk
      +dv2 : Mat = dv
      +n1T : Mat = mat_t(n1, nCt())
      dwq : Mat = mat_nt(mat_t(dq2, nCt()), n1T, 1)
      dbq : TV = mat_colsum(dq2, nCt())
      dwk : Mat = mat_nt(mat_t(dk2, nCt()), n1T, 1)
      dbk : TV = mat_colsum(dk2, nCt())
      dwv : Mat = mat_nt(mat_t(dv2, nCt()), n1T, 1)
      dbv : TV = mat_colsum(dv2, nCt())
      dn1 : Mat = mat_add(mat_add(mat_nt(dq2, wqT, kC()), mat_nt(dk2, wkT, kC())), mat_nt(dv2, wvT, kC()))
      +dn1b : Mat = dn1
      dln1g : TV = mat_colsum(mat_mul(dn1b, xh1), nCt())
      dln1b : TV = mat_colsum(dn1b, nCt())
      +de : Mat = mat_add(dr1, mat_ln_bwd(dn1b, xh1, rs1, ln1g))
      dwpe : Mat = de
      dwte : Mat = mat_add(dwte_head, scatter_add(toks2, de, mat_zeros(nV(), nCt())))
      G{loss, dwte, dwpe, dln1g, dln1b, dwq, dbq, dwk, dbk, dwv, dbv, dwo, dbo, dln2g, dln2b, dwfc, dbfc, dwmp, dbmp, dlnfg, dlnfb}

# @@GENERATED@@

# @@BATCH@@

# --- Training loop: pure, parameters / gradients / Adam state stay as tile records (G) --------------------------
type TrainOut is Data:
  TO{losses: Vec}

def train_pure(+n: Nat, +p: G, +m: G, +v: G, +step: U32, losses: Vec) -> TrainOut:
  match n:
    case 0n:
      TO{vrev(losses)}
    case 1n+k:
      +g : G = batch_grad(depth(), make_p(p), 0)
      +loss : F32 = (g_loss(g) * inv_bt() : F32)
      +tf : F32 = U32.to_f32((step + 1 : U32))
      c1 : F32 = (1.0 - F32.pow(adam_b1(), tf) : F32)
      c2 : F32 = (1.0 - F32.pow(adam_b2(), tf) : F32)
      +m2 : G = g_adam_m(g, m, inv_bt(), adam_b1(), (1.0 - adam_b1() : F32))
      +v2 : G = g_adam_v(g, v, inv_bt(), adam_b2(), (1.0 - adam_b2() : F32))
      +p2 : G = g_adam_p(p, m2, v2, adam_lr(), c1, c2, adam_eps())
      train_pure(k, p2, m2, v2, (step + 1 : U32), VCon{loss, losses})

def train_gpu(+n: Nat) -> TrainOut:
  train_pure!(n, g_of_flat(init_flat()), g_of_flat(vec_zeros(n_params())), g_of_flat(vec_zeros(n_params())), 0, VNil{})

def print_losses(+ls: Vec, +i: U32) -> IO(Unit):
  match ls:
    case VNil{}:
      do IO<Unit>:
        return Unit{}
    case VCon{+l, t}:
      do IO<Unit>:
        IO.print("step " ++ U32.show(i) ++ " loss " ++ F32.show(l) ++ " t_ms 0")
        print_losses(t, (i + 1 : U32))

def print_out(o: TrainOut) -> IO(Unit):
  match o:
    case TO{ls}:
      print_losses(ls, 0)

def steps_of(m: Maybe<&2, U32>) -> Nat:
  match m:
    case None{}:
      n_steps()
    case Some{+k}:
      U32.to_nat(k)

def parse_steps(+s: String) -> Nat:
  steps_of(U32.read(s))

def steps_arg(args: List<String>) -> Nat:
  match args:
    case Con{_, Con{s, _}}:
      parse_steps(s)
    case _:
      n_steps()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    +n : Nat = steps_arg(args)
    +t0 : Nat <- IO.now()
    out : TrainOut = train_gpu(n)
    print_out(out)
    t1 : Nat <- IO.now()
    IO.print("total_ms " ++ Nat.show(Nat.sub(t1, t0)))
