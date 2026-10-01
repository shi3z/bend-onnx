# ==============================================================================
# nanoGPT TRAINING in Pure Bend: forward + hand-written backward + AdamW(wd=0)
#
# Prepended by the generator (nanogpt/bench_train.py):
#   - src/matrix.bend primitives (Vec, vec_dot, vec_softmax, ...)
#   - size constants: nV() nC() nF() nT(), sz_*(), fscale()
# Appended by the generator: init weights, phrases, main().
#
# Architecture == Karpathy nanoGPT (1 layer, 1 head, pre-LN, weight tying,
# tanh-GELU), operating on whole sequences as matrices (list of row Vecs).
# Every gradient is derived by hand from three products:
#   A B^T  (mat_nt)   A B  (mat_nn)   A^T B  (mat_tn)
# ==============================================================================

type Mat is Data:
  MNil{}
  MCon{row: Vec, rest: Mat}

type Toks is Data:
  TNil{}
  TCon{tok: U32, rest: Toks}

# --- Vec helpers ---------------------------------------------------------------
# NOTE: a GPU lane's continuation stack is only ~2k frames deep, so anything that walks a
# LONG vector (the flat parameter / gradient vector) is written as a tail-recursive
# accumulate-reversed loop (compiles to a flat native loop) instead of `VCon{h, f(t)}`.
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

def vflat_add_acc(a: Vec, b: Vec, acc: Vec) -> Vec:
  match a b:
    case VCon{+ah, at} VCon{+bh, bt}:
      vflat_add_acc(at, bt, VCon{(ah + bh : F32), acc})
    case _ _:
      acc

def vflat_add(a: Vec, b: Vec) -> Vec:
  vrev(vflat_add_acc(a, b, VNil{}))

def vflat_scale_acc(+s: F32, v: Vec, acc: Vec) -> Vec:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vflat_scale_acc(s, t, VCon{(s * h : F32), acc})

def vflat_scale(+s: F32, v: Vec) -> Vec:
  vrev(vflat_scale_acc(s, v, VNil{}))

def vec_add_scalar(+s: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(h + s : F32), vec_add_scalar(s, t)}

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

# --- Mat helpers -----------------------------------------------------------------
def mat_zeros(+r: Nat, +c: Nat) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MCon{vec_zeros(c), mat_zeros(p, c)}

def mat_of(+r: Nat, +c: Nat, +v: Vec) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MCon{vtake(c, v), mat_of(p, c, vdrop(c, v))}

def mat_flat(m: Mat) -> Vec:
  match m:
    case MNil{}:
      VNil{}
    case MCon{r, rt}:
      vcat(r, mat_flat(rt))

def mat_add(a: Mat, b: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      MCon{vec_add(ar, br), mat_add(at, bt)}
    case _ _:
      MNil{}

def mat_mul(a: Mat, b: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      MCon{vec_mul(ar, br), mat_mul(at, bt)}
    case _ _:
      MNil{}

def mat_scale(+s: F32, m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{vec_scale(s, r), mat_scale(s, rt)}

def mat_add_bias(m: Mat, +b: Vec) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{vec_add(r, b), mat_add_bias(rt, b)}

def mat_colsum_acc(m: Mat, acc: Vec) -> Vec:
  match m:
    case MNil{}:
      acc
    case MCon{r, rt}:
      mat_colsum_acc(rt, vec_add(acc, r))

def mat_colsum(m: Mat, +w: Nat) -> Vec:
  mat_colsum_acc(m, vec_zeros(w))

# A B^T : rows of A dotted with rows of B
def mat_nt_row(+a: Vec, b: Mat) -> Vec:
  match b:
    case MNil{}:
      VNil{}
    case MCon{br, bt}:
      VCon{vec_dot(a, br), mat_nt_row(a, bt)}

def mat_nt(a: Mat, +b: Mat) -> Mat:
  match a:
    case MNil{}:
      MNil{}
    case MCon{ar, at}:
      MCon{mat_nt_row(ar, b), mat_nt(at, b)}

# A B : each row of A is a linear combination of the rows of B (w = width of B)
def lincomb(r: Vec, b: Mat, acc: Vec) -> Vec:
  match r b:
    case VCon{+c, rt} MCon{br, bt}:
      lincomb(rt, bt, vec_add(acc, vec_scale(c, br)))
    case _ _:
      acc

def mat_nn(a: Mat, +b: Mat, +w: Nat) -> Mat:
  match a:
    case MNil{}:
      MNil{}
    case MCon{ar, at}:
      MCon{lincomb(ar, b, vec_zeros(w)), mat_nn(at, b, w)}

# A^T B = sum_t outer(A_t, B_t)   (ra = width of A, cb = width of B)
def mat_outer_acc(acc: Mat, a: Vec, +b: Vec) -> Mat:
  match acc a:
    case MCon{row, rest} VCon{+ai, at}:
      MCon{vec_add(row, vec_scale(ai, b)), mat_outer_acc(rest, at, b)}
    case _ _:
      acc

def mat_tn_acc(a: Mat, b: Mat, acc: Mat) -> Mat:
  match a b:
    case MCon{ar, at} MCon{br, bt}:
      mat_tn_acc(at, bt, mat_outer_acc(acc, ar, br))
    case _ _:
      acc

def mat_tn(a: Mat, b: Mat, +ra: Nat, +cb: Nat) -> Mat:
  mat_tn_acc(a, b, mat_zeros(ra, cb))

# acc[idx] += row
def mat_row_add(acc: Mat, +idx: U32, row: Vec) -> Mat:
  match acc idx:
    case MCon{r, rt} 0:
      MCon{vec_add(r, row), rt}
    case MCon{r, rt} _:
      MCon{r, mat_row_add(rt, U32.sub(idx, 1), row)}
    case _ _:
      acc

def mat_row_get(+m: Mat, +idx: U32) -> Vec:
  match m idx:
    case MCon{+r, _} 0:
      r
    case MCon{_, rt} _:
      mat_row_get(rt, U32.sub(idx, 1))
    case _ _:
      VNil{}

# --- Embedding ---------------------------------------------------------------------
def embed(+wte: Mat, toks: Toks, wpe: Mat) -> Mat:
  match toks wpe:
    case TCon{+tk, tt} MCon{pr, pt}:
      MCon{vec_add(mat_row_get(wte, tk), pr), embed(wte, tt, pt)}
    case _ _:
      MNil{}

# d wte[tok_t] += de_t
def scatter_add(toks: Toks, de: Mat, acc: Mat) -> Mat:
  match toks de:
    case TCon{+tk, tt} MCon{dr, dt}:
      scatter_add(tt, dt, mat_row_add(acc, tk, dr))
    case _ _:
      acc

# --- GELU (tanh approximation) ------------------------------------------------------
def mat_gelu(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{vec_gelu(r), mat_gelu(rt)}

def f32_gelu_grad(+x: F32) -> F32:
  +c : F32 = 0.79788456
  +x2 : F32 = (x * x : F32)
  inner : F32 = (c * (x + 0.044715 * x * x2 : F32) : F32)
  +t : F32 = F32.tanh(inner)
  sech2 : F32 = (1.0 - t * t : F32)
  (0.5 * (1.0 + t : F32) + 0.5 * x * sech2 * c * (1.0 + 0.134145 * x2 : F32) : F32)

def vec_gelu_bwd(f: Vec, dh: Vec) -> Vec:
  match f dh:
    case VCon{+fh, ft} VCon{+dv, dt}:
      VCon{(dv * f32_gelu_grad(fh) : F32), vec_gelu_bwd(ft, dt)}
    case _ _:
      VNil{}

def mat_gelu_bwd(f: Mat, dh: Mat) -> Mat:
  match f dh:
    case MCon{fr, ft} MCon{dr, dt}:
      MCon{vec_gelu_bwd(fr, dr), mat_gelu_bwd(ft, dt)}
    case _ _:
      MNil{}

# --- LayerNorm over rows (fwd cache = xhat, rstd) ---------------------------------------
def row_rstd(+v: Vec) -> F32:
  +m : F32 = vec_mean(v)
  var : F32 = vec_var(v, m)
  (1.0 / F32.sqrt((var + 0.00001 : F32)) : F32)

def row_xhat(+v: Vec) -> Vec:
  +m : F32 = vec_mean(v)
  r : F32 = row_rstd(v)
  vec_scale(r, vec_add_scalar(F32.neg(m), v))

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

def ln_affine(xhat: Mat, +g: Vec, +b: Vec) -> Mat:
  match xhat:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{vec_add(vec_mul(r, g), b), ln_affine(rt, g, b)}

# dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),  dxhat = dy * gamma
def row_ln_bwd(dy: Vec, +xhat: Vec, +rstd: F32, +g: Vec) -> Vec:
  +dxh : Vec = vec_mul(dy, g)
  +m1 : F32 = vec_mean(dxh)
  +m2 : F32 = vec_mean(vec_mul(dxh, xhat))
  vec_scale(rstd, vec_sub(vec_add_scalar(F32.neg(m1), dxh), vec_scale(m2, xhat)))

def mat_ln_bwd(dy: Mat, xhat: Mat, rstd: Vec, +g: Vec) -> Mat:
  match dy xhat rstd:
    case MCon{d, dt} MCon{xh, xt} VCon{+r, rt}:
      MCon{row_ln_bwd(d, xh, r, g), mat_ln_bwd(dt, xt, rt, g)}
    case _ _ _:
      MNil{}

# --- Causal attention pieces ------------------------------------------------------------
def mask_row(+v: Vec, +lim: U32, +i: U32) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      keep : Bool = U32.is_le(i, lim)
      neg_inf : F32 = F32.neg(1000000000.0)
      VCon{Bool.pick(F32, keep, h, neg_inf), mask_row(t, lim, (i + 1 : U32))}

def mat_mask(m: Mat, +t: U32) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{mask_row(r, t, 0), mat_mask(rt, (t + 1 : U32))}

def mat_softmax(m: Mat) -> Mat:
  match m:
    case MNil{}:
      MNil{}
    case MCon{r, rt}:
      MCon{vec_softmax(r), mat_softmax(rt)}

# dS = A * (dA - rowsum(A * dA))
def row_softmax_bwd(+a: Vec, +da: Vec) -> Vec:
  +d : F32 = vec_dot(a, da)
  vec_mul(a, vec_add_scalar(F32.neg(d), da))

def mat_softmax_bwd(a: Mat, da: Mat) -> Mat:
  match a da:
    case MCon{ar, at} MCon{dr, dt}:
      MCon{row_softmax_bwd(ar, dr), mat_softmax_bwd(at, dt)}
    case _ _:
      MNil{}

# --- Cross entropy ---------------------------------------------------------------------------
def row_ce(+p: Vec, +y: U32) -> F32:
  pv : F32 = F32.max(vec_get(p, y), 0.000000000000000000000000000001)
  F32.neg(F32.log(pv))

def ce_loss(pr: Mat, tg: Toks, +acc: F32) -> F32:
  match pr tg:
    case MCon{pv, pt} TCon{+y, yt}:
      ce_loss(pt, yt, (acc + row_ce(pv, y) : F32))
    case _ _:
      acc

# d logits = softmax - onehot(y)
def row_ce_grad(p: Vec, +y: U32, +i: U32) -> Vec:
  match p:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      hit : Bool = U32.is_eq(i, y)
      g : F32 = Bool.pick(F32, hit, (h - 1.0 : F32), h)
      VCon{g, row_ce_grad(t, y, (i + 1 : U32))}

def mat_ce_grad(pr: Mat, tg: Toks) -> Mat:
  match pr tg:
    case MCon{pv, pt} TCon{+y, yt}:
      MCon{row_ce_grad(pv, y, 0), mat_ce_grad(pt, yt)}
    case _ _:
      MNil{}

# --- Parameters (flat Vec <-> record) ------------------------------------------------------------
type P is Data:
  P{wte: Mat, wpe: Mat, ln1g: Vec, ln1b: Vec,
    wq: Mat, bq: Vec, wk: Mat, bk: Vec, wv: Mat, bv: Vec, wo: Mat, bo: Vec,
    ln2g: Vec, ln2b: Vec, wfc: Mat, bfc: Vec, wmp: Mat, bmp: Vec,
    lnfg: Vec, lnfb: Vec}

def lin_fwd(+x: Mat, +w: Mat, +b: Vec) -> Mat:
  mat_add_bias(mat_nt(x, w), b)

# --- One sample: forward + backward -> [loss, d params ...] ------------------------------------
def sample_grad(+p: P, toks: Toks, tgts: Toks) -> Vec:
  match p:
    case P{+wte, +wpe, +ln1g, +ln1b, +wq, +bq, +wk, +bk, +wv, +bv, +wo, +bo, +ln2g, +ln2b, +wfc, +bfc, +wmp, +bmp, +lnfg, +lnfb}:
      +toks2 : Toks = toks
      # ---- forward ----
      +e : Mat = embed(wte, toks2, wpe)
      +xh1 : Mat = mat_xhat(e)
      +rs1 : Vec = mat_rstd(e)
      +n1 : Mat = ln_affine(xh1, ln1g, ln1b)
      +q : Mat = lin_fwd(n1, wq, bq)
      +k : Mat = lin_fwd(n1, wk, bk)
      +v : Mat = lin_fwd(n1, wv, bv)
      +a : Mat = mat_softmax(mat_mask(mat_scale(fscale(), mat_nt(q, k)), 0))
      +ctx : Mat = mat_nn(a, v, nC())
      +o : Mat = lin_fwd(ctx, wo, bo)
      +r1 : Mat = mat_add(e, o)
      +xh2 : Mat = mat_xhat(r1)
      +rs2 : Vec = mat_rstd(r1)
      +n2 : Mat = ln_affine(xh2, ln2g, ln2b)
      +f : Mat = lin_fwd(n2, wfc, bfc)
      +h : Mat = mat_gelu(f)
      +mo : Mat = lin_fwd(h, wmp, bmp)
      +r2 : Mat = mat_add(r1, mo)
      +xhf : Mat = mat_xhat(r2)
      +rsf : Vec = mat_rstd(r2)
      +nf : Mat = ln_affine(xhf, lnfg, lnfb)
      +logits : Mat = mat_nt(nf, wte)
      +pr : Mat = mat_softmax(logits)
      +tg : Toks = tgts
      loss : F32 = ce_loss(pr, tg, 0.0)
      # ---- backward ----
      +dlog : Mat = mat_ce_grad(pr, tg)
      dwte_head : Mat = mat_tn(dlog, nf, nV(), nC())
      +dnf : Mat = mat_nn(dlog, wte, nC())
      dlnfg : Vec = mat_colsum(mat_mul(dnf, xhf), nC())
      dlnfb : Vec = mat_colsum(dnf, nC())
      +dr2 : Mat = mat_ln_bwd(dnf, xhf, rsf, lnfg)
      dwmp : Mat = mat_tn(dr2, h, nC(), nF())
      dbmp : Vec = mat_colsum(dr2, nC())
      +dh : Mat = mat_nn(dr2, wmp, nF())
      +df : Mat = mat_gelu_bwd(f, dh)
      dwfc : Mat = mat_tn(df, n2, nF(), nC())
      dbfc : Vec = mat_colsum(df, nF())
      +dn2 : Mat = mat_nn(df, wfc, nC())
      dln2g : Vec = mat_colsum(mat_mul(dn2, xh2), nC())
      dln2b : Vec = mat_colsum(dn2, nC())
      +dr1 : Mat = mat_add(dr2, mat_ln_bwd(dn2, xh2, rs2, ln2g))
      dwo : Mat = mat_tn(dr1, ctx, nC(), nC())
      dbo : Vec = mat_colsum(dr1, nC())
      +dctx : Mat = mat_nn(dr1, wo, nC())
      +da : Mat = mat_nt(dctx, v)
      dv : Mat = mat_tn(a, dctx, nT(), nC())
      +ds : Mat = mat_softmax_bwd(a, da)
      dq : Mat = mat_scale(fscale(), mat_nn(ds, k, nC()))
      dk : Mat = mat_scale(fscale(), mat_tn(ds, q, nT(), nC()))
      +dq2 : Mat = dq
      +dk2 : Mat = dk
      +dv2 : Mat = dv
      dwq : Mat = mat_tn(dq2, n1, nC(), nC())
      dbq : Vec = mat_colsum(dq2, nC())
      dwk : Mat = mat_tn(dk2, n1, nC(), nC())
      dbk : Vec = mat_colsum(dk2, nC())
      dwv : Mat = mat_tn(dv2, n1, nC(), nC())
      dbv : Vec = mat_colsum(dv2, nC())
      dn1 : Mat = mat_add(mat_add(mat_nn(dq2, wq, nC()), mat_nn(dk2, wk, nC())), mat_nn(dv2, wv, nC()))
      +dn1b : Mat = dn1
      dln1g : Vec = mat_colsum(mat_mul(dn1b, xh1), nC())
      dln1b : Vec = mat_colsum(dn1b, nC())
      +de : Mat = mat_add(dr1, mat_ln_bwd(dn1b, xh1, rs1, ln1g))
      dwpe : Mat = de
      dwte : Mat = mat_add(dwte_head, scatter_add(toks2, de, mat_zeros(nV(), nC())))
      vcat(VCon{loss, VNil{}}, vcat(mat_flat(dwte), vcat(mat_flat(dwpe),
        vcat(dln1g, vcat(dln1b,
        vcat(mat_flat(dwq), vcat(dbq, vcat(mat_flat(dwk), vcat(dbk,
        vcat(mat_flat(dwv), vcat(dbv, vcat(mat_flat(dwo), vcat(dbo,
        vcat(dln2g, vcat(dln2b,
        vcat(mat_flat(dwfc), vcat(dbfc, vcat(mat_flat(dwmp), vcat(dbmp,
        vcat(dlnfg, dlnfb))))))))))))))))))))

# @@GENERATED@@

# --- Batch: balanced fork-join tree of 2^d samples, gradients summed on the way up --------------
def batch_grad(+d: Nat, +p: P, +idx: U32) -> Vec:
  match d:
    case 0n:
      sample_grad(p, phrase_x(idx), phrase_y(idx))
    case 1n+k:
      a b = batch_grad(k, p, (idx * 2 : U32)) batch_grad(k, p, (idx * 2 + 1 : U32))
      vflat_add(a, b)

def batch_grad_gpu(+d: Nat, +p: P) -> Vec:
  batch_grad!(d, p, 0)

# --- AdamW (weight_decay = 0) over the flat parameter vector ---------------------------------------
def adam_m_acc(g: Vec, m: Vec, +b1: F32, acc: Vec) -> Vec:
  match g m:
    case VCon{+gh, gt} VCon{+mh, mt}:
      adam_m_acc(gt, mt, b1, VCon{(b1 * mh + (1.0 - b1 : F32) * gh : F32), acc})
    case _ _:
      acc

def adam_m(g: Vec, m: Vec, +b1: F32) -> Vec:
  vrev(adam_m_acc(g, m, b1, VNil{}))

def adam_v_acc(g: Vec, v: Vec, +b2: F32, acc: Vec) -> Vec:
  match g v:
    case VCon{+gh, gt} VCon{+vh, vt}:
      adam_v_acc(gt, vt, b2, VCon{(b2 * vh + (1.0 - b2 : F32) * gh * gh : F32), acc})
    case _ _:
      acc

def adam_v(g: Vec, v: Vec, +b2: F32) -> Vec:
  vrev(adam_v_acc(g, v, b2, VNil{}))

def adam_p_acc(p: Vec, m: Vec, v: Vec, +lr: F32, +c1: F32, +c2: F32, +eps: F32, acc: Vec) -> Vec:
  match p m v:
    case VCon{+ph, pt} VCon{+mh, mt} VCon{+vh, vt}:
      den : F32 = (F32.sqrt((vh / c2 : F32)) + eps : F32)
      adam_p_acc(pt, mt, vt, lr, c1, c2, eps, VCon{(ph - lr * (mh / c1 : F32) / den : F32), acc})
    case _ _ _:
      acc

def adam_p(p: Vec, m: Vec, v: Vec, +lr: F32, +c1: F32, +c2: F32, +eps: F32) -> Vec:
  vrev(adam_p_acc(p, m, v, lr, c1, c2, eps, VNil{}))

# --- Training loop -------------------------------------------------------------------------------------
def print_step(+flag: Bool, +step: U32, +loss: F32, +t0: Nat) -> IO(Unit):
  match flag:
    case True{}:
      do IO<Unit>:
        t : Nat <- IO.now()
        IO.print("step " ++ U32.show(step) ++ " loss " ++ F32.show(loss) ++ " t_ms " ++ Nat.show(Nat.sub(t, t0)))
    case False{}:
      do IO<Unit>:
        return Unit{}

def train_loop(+n: Nat, +flat: Vec, +m: Vec, +v: Vec, +step: U32, +t0: Nat) -> IO(Unit):
  match n:
    case 0n:
      do IO<Unit>:
        return Unit{}
    case 1n+k:
      do IO<Unit>:
        +g : Vec = vflat_scale(inv_bt(), batch_grad_gpu(depth(), unflatten(flat)))
        +loss : F32 = vec_get(g, 0)
        +grads : Vec = vdrop(1n, g)
        +tf : F32 = U32.to_f32((step + 1 : U32))
        c1 : F32 = (1.0 - F32.pow(adam_b1(), tf) : F32)
        c2 : F32 = (1.0 - F32.pow(adam_b2(), tf) : F32)
        +m2 : Vec = adam_m(grads, m, adam_b1())
        +v2 : Vec = adam_v(grads, v, adam_b2())
        +flat2 : Vec = adam_p(flat, m2, v2, adam_lr(), c1, c2, adam_eps())
        print_step(U32.is_eq(U32.mod(step, print_every()), 0), step, loss, t0)
        train_loop(k, flat2, m2, v2, (step + 1 : U32), t0)

def main() -> IO(Unit):
  do IO<Unit>:
    +t0 : Nat <- IO.now()
    IO.print("config B " ++ U32.show(batch_size()) ++ " C " ++ Nat.show(nC()))
    train_loop(n_steps(), init_flat(), vec_zeros(n_params()), vec_zeros(n_params()), 0, t0)
    t1 : Nat <- IO.now()
    IO.print("total_ms " ++ Nat.show(Nat.sub(t1, t0)))
