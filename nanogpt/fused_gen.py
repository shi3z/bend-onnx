"""
v4: fused, register-resident token kernels (C=16, F=64, V=32, T<=16).

Why (measured, see README): a chain of 16-wide tile ops whose intermediates stay in registers runs
at ~13 G tile-ops/s on the A100, the same ops as list/matrix nodes on the heap at ~0.7 G/s. A single
flat function doing the whole tail of the model for one token (LN, fc, GELU, proj, LN, logits,
softmax, loss; ~160 tile dot products) reached ~75 M tokens/s on the GPU, 8x the 24-core CPU. So
the training step is rewritten as a few such fused functions per token, with only the tiles that
the backward pass really needs stored on the heap.

Per sample (one GPU lane):
  pass 1  tokens in order: embed, LN1, qkv, attention over the keys seen so far, o-proj, LN2, MLP,
          LN_f, logits, softmax loss, then the whole token-local backward (dlogits ... dr1, dctx, ds, dq)
  pass 2  keys: dk_j, dv_j from the stored attention rows
  pass 3  tokens: dn1 -> LN1 backward -> de
  pass 4  weight gradients: register accumulators looping over the sample's tokens
"""
from tile_gen import tile_ops, IDX

I16 = list(IDX)


def nest(con, items, tail="_"):
    s = tail
    for x in reversed(items):
        s = f"{con}{{{x}, {s}}}"
    return s


def tile_extras() -> str:
    """lane access and a few helpers on top of tile_gen.tile_ops(False)."""
    return """# broadcast, lane read / write (all branch-free, register only)
def t_bcast(+s: F32) -> T16:
  T16{s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s}

def t_get(+a: T16, +i: U32) -> F32:
  t_dot(a, t_onehot(i))

def t_set(+a: T16, +i: U32, +v: F32) -> T16:
  +oh : T16 = t_onehot(i)
  t_add(t_sub(a, t_mul(a, oh)), t_scale(v, oh))

def t_neginf() -> T16:
  t_bcast(F32.neg(1000000000.0))

type Tl is Data:
  TlN{}
  TlC{h: T16, t: Tl}

# four accumulator tiles (one pass of a weight-gradient loop covers four rows)
type R4t is Data:
  R4t{a: T16, b: T16, c: T16, d: T16}

def r4t_rows(rs: R4t, rows: Mat) -> Mat:
  match rs:
    case R4t{r0, r1, r2, r3}:
      MC1{r0, MC1{r1, MC1{r2, MC1{r3, rows}}}}

def lane0(+a: T16) -> F32:
  match a:
    case T16{+x, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      x

def lane1(+a: T16) -> F32:
  match a:
    case T16{_, +x, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      x

def lane2(+a: T16) -> F32:
  match a:
    case T16{_, _, +x, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      x

def lane3(+a: T16) -> F32:
  match a:
    case T16{_, _, _, +x, _, _, _, _, _, _, _, _, _, _, _, _}:
      x

def lane4(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, +x, _, _, _, _, _, _, _, _, _, _, _}:
      x

def lane5(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, +x, _, _, _, _, _, _, _, _, _, _}:
      x

def lane6(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, +x, _, _, _, _, _, _, _, _, _}:
      x

def lane7(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, +x, _, _, _, _, _, _, _, _}:
      x

def lane8(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, +x, _, _, _, _, _, _, _}:
      x

def lane9(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, +x, _, _, _, _, _, _}:
      x

def lane10(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, +x, _, _, _, _, _}:
      x

def lane11(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, _, +x, _, _, _, _}:
      x

def lane12(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, _, _, +x, _, _, _}:
      x

def lane13(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, _, _, _, +x, _, _}:
      x

def lane14(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, _, _, _, _, +x, _}:
      x

def lane15(+a: T16) -> F32:
  match a:
    case T16{_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, +x}:
      x

def tl_rev_onto(x: Tl, acc: Tl) -> Tl:
  match x:
    case TlN{}:
      acc
    case TlC{h, t}:
      tl_rev_onto(t, TlC{h, acc})

def tl_rev(x: Tl) -> Tl:
  tl_rev_onto(x, TlN{})
"""


def mv_kernels() -> str:
    """y = W x for 16-row blocks held as Mat cells (MC1 / MC2 / MC4); ONE tile result."""
    bs = [f"b{i}" for i in I16]
    o = []
    o.append(f"""def mv1(+x: T16, w: Mat) -> T16:
  match w:
    case {nest('MC1', bs)}:
      T16{{{', '.join(f't_dot(x, b{i})' for i in I16)}}}
    case _:
      x
""")
    rows2 = [f"b{i}a, b{i}b" for i in I16]
    s = "_"
    for i in reversed(I16):
        s = f"MC2{{b{i}a, b{i}b, {s}}}"
    o.append(f"""def mv2(+x0: T16, +x1: T16, w: Mat) -> T16:
  match w:
    case {s}:
      T16{{{', '.join(f'(t_dot(x0, b{i}a) + t_dot(x1, b{i}b) : F32)' for i in I16)}}}
    case _:
      x0
""")
    s = "_"
    for i in reversed(I16):
        s = f"MC4{{b{i}a, b{i}b, b{i}c, b{i}d, {s}}}"
    o.append(f"""def dot4s(+h0: T16, +h1: T16, +h2: T16, +h3: T16, a: T16, b: T16, c: T16, d: T16) -> F32:
  ((t_dot(h0, a) + t_dot(h1, b)) + (t_dot(h2, c) + t_dot(h3, d)) : F32)

def mv4(+h0: T16, +h1: T16, +h2: T16, +h3: T16, w: Mat) -> T16:
  match w:
    case {s}:
      T16{{{', '.join(f'dot4s(h0, h1, h2, h3, b{i}a, b{i}b, b{i}c, b{i}d)' for i in I16)}}}
    case _:
      h0
""")
    return "\n".join(o)


def ln_ops() -> str:
    return """def ln_xh(+x: T16) -> T16:
  +m : F32 = (t_sum(x) * inv_c() : F32)
  +c : T16 = t_addc(F32.neg(m), x)
  v : F32 = (t_dot(c, c) * inv_c() : F32)
  t_scale((1.0 / F32.sqrt((v + 0.00001 : F32)) : F32), c)

def ln_rs(+x: T16) -> F32:
  +m : F32 = (t_sum(x) * inv_c() : F32)
  +c : T16 = t_addc(F32.neg(m), x)
  v : F32 = (t_dot(c, c) * inv_c() : F32)
  (1.0 / F32.sqrt((v + 0.00001 : F32)) : F32)

# dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),  dxhat = dy * gamma
def ln_bwd(+dy: T16, +g: T16, +xh: T16, +rs: F32) -> T16:
  +dxh : T16 = t_mul(dy, g)
  +m1 : F32 = (t_sum(dxh) * inv_c() : F32)
  +m2 : F32 = (t_dot(dxh, xh) * inv_c() : F32)
  t_scale(rs, t_sub(t_addc(F32.neg(m1), dxh), t_scale(m2, xh)))

# softmax over a tile of 16 lanes
def t_softmax(+s: T16) -> T16:
  +mx : F32 = t_max(s)
  +e : T16 = t_expm(mx, s)
  t_scale((1.0 / t_sum(e) : F32), e)

def tv0(tv: TV) -> T16:
  match tv:
    case QCon{h, _}:
      h
    case _:
      t_zero()
"""


# ------------------------------------------------------------------------------------------------
# weights record Q: everything the fused kernels read, pre-split into 16-row blocks (once per step)
# ------------------------------------------------------------------------------------------------
QF = ["wteM", "wte0", "wte1", "wpe", "g1", "b1", "g2", "b2", "gf", "bf",
      "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
      "wfc0", "wfc1", "wfc2", "wfc3", "bfc0", "bfc1", "bfc2", "bfc3", "wmp", "bmp",
      "wteT", "wmpT0", "wmpT1", "wmpT2", "wmpT3", "wfcT", "woT", "wqT", "wkT", "wvT"]
QT = {n: ("T16" if n in ("g1", "b1", "g2", "b2", "gf", "bf", "bq", "bk", "bv", "bo", "bmp",
                          "bfc0", "bfc1", "bfc2", "bfc3") else "Mat") for n in QF}


def qpat(*used):
    """pattern for Q that binds only `used` (as reusable +name), `_` elsewhere"""
    return "Q{" + ", ".join(f"+{n}" if n in used else "_" for n in QF) + "}"


def q_type() -> str:
    return "type Q is Data:\n  Q{" + ", ".join(f"{n}: {QT[n]}" for n in QF) + "}\n"


def make_q() -> str:
    """G (params as Mat/TV) -> Q. Runs once per step on a single lane."""
    return """def tv_nth(tv: TV, +m: U32) -> T16:
  match tv m:
    case QCon{h, _} 0:
      h
    case QCon{_, t} _:
      tv_nth(t, U32.sub(m, 1))
    case _ _:
      t_zero()

def make_q(p: G) -> Q:
  match p:
    case G{_, +wte, +wpe, +ln1g, +ln1b, +wq, +bq, +wk, +bk, +wv, +bv, +wo, +bo, +ln2g, +ln2b, +wfc, +bfc, +wmp, +bmp, +lnfg, +lnfb}:
      +wte0 : Mat = mat_take(wte, 16)
      +wte1 : Mat = mat_take(mat_drop(wte, 16), 16)
      +wmpT : Mat = mat_t4(wmp)
      Q{wte, wte0, wte1, wpe, tv0(ln1g), tv0(ln1b), tv0(ln2g), tv0(ln2b), tv0(lnfg), tv0(lnfb),
        wq, tv0(bq), wk, tv0(bk), wv, tv0(bv), wo, tv0(bo),
        mat_take(wfc, 16), mat_take(mat_drop(wfc, 16), 16), mat_take(mat_drop(wfc, 32), 16), mat_take(mat_drop(wfc, 48), 16),
        tv_nth(bfc, 0), tv_nth(bfc, 1), tv_nth(bfc, 2), tv_nth(bfc, 3), wmp, tv0(bmp),
        mat_zip2(mat_t1(wte0), mat_t1(wte1)),
        mat_take(wmpT, 16), mat_take(mat_drop(wmpT, 16), 16), mat_take(mat_drop(wmpT, 32), 16), mat_drop(wmpT, 48),
        mat_zip4(mat_t1(mat_take(wfc, 16)), mat_t1(mat_take(mat_drop(wfc, 16), 16)), mat_t1(mat_take(mat_drop(wfc, 32), 16)), mat_t1(mat_drop(wfc, 48))),
        mat_t1(wo), mat_t1(wq), mat_t1(wk), mat_t1(wv)}
"""


# ------------------------------------------------------------------------------------------------
# records
# ------------------------------------------------------------------------------------------------
RECS = {
    "R1": [("e", "T16"), ("xh1", "T16"), ("rs1", "F32"), ("q", "T16"), ("k", "T16"), ("v", "T16")],
    "R2": [("a", "T16"), ("ctx", "T16")],
    "R3": [("r1", "T16"), ("xh2", "T16"), ("rs2", "F32"), ("f0", "T16"), ("f1", "T16"), ("f2", "T16"), ("f3", "T16")],
    "R4": [("xhf", "T16"), ("rsf", "F32"), ("dlog0", "T16"), ("dlog1", "T16"), ("loss", "F32")],
    "R5": [("dnf", "T16"), ("dr2", "T16")],
    "R6": [("df0", "T16"), ("df1", "T16"), ("df2", "T16"), ("df3", "T16"), ("dn2", "T16"), ("dr1", "T16"), ("dctx", "T16")],
    "R7": [("ds", "T16"), ("dq", "T16")],
    "R8": [("dk", "T16"), ("dv", "T16")],
    "R9": [("dn1", "T16"), ("de", "T16")],
}


def rec_types() -> str:
    return "\n".join(f"type {r} is Data:\n  {r}{{{', '.join(f'{n}: {t}' for n, t in fs)}}}\n" for r, fs in RECS.items())


def rpat(rec, *used):
    """pattern binding only `used` fields (reusable), `_` elsewhere"""
    return f"{rec}{{" + ", ".join(f"+{n}" if n in used else "_" for n, _ in RECS[rec]) + "}"


def lpat(rec, nxt, *used):
    return f"Con{{{rpat(rec, *used)}, {nxt}}}"


# ------------------------------------------------------------------------------------------------
# stage functions: flat, register resident, one record out
# ------------------------------------------------------------------------------------------------
def stages() -> str:
    o = []
    o.append(f"""def stage1(+tk: U32, +t: U32, +w: Q) -> R1:
  match w:
    case {qpat('wteM', 'wpe', 'g1', 'b1', 'wq', 'bq', 'wk', 'bk', 'wv', 'bv')}:
      +e : T16 = t_add(nth0(wteM, tk), nth0(wpe, t))
      +xh : T16 = ln_xh(e)
      rs : F32 = ln_rs(e)
      +n1 : T16 = t_add(t_mul(xh, g1), b1)
      q : T16 = t_add(mv1(n1, wq), bq)
      k : T16 = t_add(mv1(n1, wk), bk)
      v : T16 = t_add(mv1(n1, wv), bv)
      R1{{e, xh, rs, q, k, v}}
""")
    o.append("""# scores of token t against the keys seen so far (newest first: lane j = t - index)
def sc_go(+q: T16, ks: Tl, +j: U32, +s: T16) -> T16:
  match ks:
    case TlN{}:
      s
    case TlC{k, kt}:
      sc_go(q, kt, U32.sub(j, 1), t_set(s, j, (t_dot(q, k) * fscale() : F32)))

def cx_go(+a: T16, vs: Tl, +j: U32, +acc: T16) -> T16:
  match vs:
    case TlN{}:
      acc
    case TlC{v, vt}:
      cx_go(a, vt, U32.sub(j, 1), t_add(acc, t_scale(t_get(a, j), v)))

def attn_stage(+q: T16, +ks: Tl, +vs: Tl, +t: U32) -> R2:
  +a : T16 = t_softmax(sc_go(q, ks, t, t_neginf()))
  ctx : T16 = cx_go(a, vs, t, t_zero())
  R2{a, ctx}
""")
    o.append(f"""def stage3a1(+e: T16, +ctx: T16, +w: Q) -> R3:
  match w:
    case {qpat('wo', 'bo', 'g2', 'b2', 'wfc0', 'wfc1', 'wfc2', 'wfc3', 'bfc0', 'bfc1', 'bfc2', 'bfc3')}:
      +r1 : T16 = t_add(e, t_add(mv1(ctx, wo), bo))
      +xh2 : T16 = ln_xh(r1)
      rs2 : F32 = ln_rs(r1)
      +n2 : T16 = t_add(t_mul(xh2, g2), b2)
      f0 : T16 = t_add(mv1(n2, wfc0), bfc0)
      f1 : T16 = t_add(mv1(n2, wfc1), bfc1)
      f2 : T16 = t_add(mv1(n2, wfc2), bfc2)
      f3 : T16 = t_add(mv1(n2, wfc3), bfc3)
      R3{{r1, xh2, rs2, f0, f1, f2, f3}}
""")
    o.append(f"""def stage3a2(+r1: T16, +f0: T16, +f1: T16, +f2: T16, +f3: T16, +y: U32, +w: Q) -> R4:
  match w:
    case {qpat('wmp', 'bmp', 'gf', 'bf', 'wte0', 'wte1')}:
      +r2 : T16 = t_add(r1, t_add(mv4(t_gelu(f0), t_gelu(f1), t_gelu(f2), t_gelu(f3), wmp), bmp))
      +xhf : T16 = ln_xh(r2)
      rsf : F32 = ln_rs(r2)
      +nf : T16 = t_add(t_mul(xhf, gf), bf)
      +l0 : T16 = mv1(nf, wte0)
      +l1 : T16 = mv1(nf, wte1)
      +mx : F32 = F32.max(t_max(l0), t_max(l1))
      +e0 : T16 = t_expm(mx, l0)
      +e1 : T16 = t_expm(mx, l1)
      +inv : F32 = (1.0 / (t_sum(e0) + t_sum(e1) : F32) : F32)
      +p0 : T16 = t_scale(inv, e0)
      +p1 : T16 = t_scale(inv, e1)
      +oh0 : T16 = t_onehot(y)
      +oh1 : T16 = t_onehot(U32.sub(y, 16))
      py : F32 = (t_dot(p0, oh0) + t_dot(p1, oh1) : F32)
      loss : F32 = F32.neg(F32.log(F32.max(py, 0.000000000000000000000000000001)))
      R4{{xhf, rsf, t_sub(p0, oh0), t_sub(p1, oh1), loss}}
""")
    o.append(f"""def stage3b1(+dlog0: T16, +dlog1: T16, +xhf: T16, +rsf: F32, +w: Q) -> R5:
  match w:
    case {qpat('wteT', 'gf')}:
      +dnf : T16 = mv2(dlog0, dlog1, wteT)
      R5{{dnf, ln_bwd(dnf, gf, xhf, rsf)}}
""")
    o.append(f"""def stage3b2(+dr2: T16, +f0: T16, +f1: T16, +f2: T16, +f3: T16, +xh2: T16, +rs2: F32, +w: Q) -> R6:
  match w:
    case {qpat('wmpT0', 'wmpT1', 'wmpT2', 'wmpT3', 'wfcT', 'woT', 'g2')}:
      +df0 : T16 = t_gelu_bwd(f0, mv1(dr2, wmpT0))
      +df1 : T16 = t_gelu_bwd(f1, mv1(dr2, wmpT1))
      +df2 : T16 = t_gelu_bwd(f2, mv1(dr2, wmpT2))
      +df3 : T16 = t_gelu_bwd(f3, mv1(dr2, wmpT3))
      +dn2 : T16 = mv4(df0, df1, df2, df3, wfcT)
      +dr1 : T16 = t_add(dr2, ln_bwd(dn2, g2, xh2, rs2))
      R6{{df0, df1, df2, df3, dn2, dr1, mv1(dr1, woT)}}
""")
    o.append("""# attention backward local to token t: ds = a * (da - sum(a * da)), dq = scale * sum_j ds_j k_j
def da_go(+dctx: T16, vs: Tl, +j: U32, +da: T16) -> T16:
  match vs:
    case TlN{}:
      da
    case TlC{v, vt}:
      da_go(dctx, vt, U32.sub(j, 1), t_set(da, j, t_dot(dctx, v)))

def dq_go(+ds: T16, ks: Tl, +j: U32, +acc: T16) -> T16:
  match ks:
    case TlN{}:
      acc
    case TlC{k, kt}:
      dq_go(ds, kt, U32.sub(j, 1), t_add(acc, t_scale((t_get(ds, j) * fscale() : F32), k)))

def attn_bwd(+a: T16, +dctx: T16, +ks: Tl, +vs: Tl, +t: U32) -> R7:
  +da : T16 = da_go(dctx, vs, t, t_zero())
  +ds : T16 = t_mul(a, t_addc(F32.neg(t_dot(a, da)), da))
  R7{ds, dq_go(ds, ks, t, t_zero())}
""")
    o.append(f"""def stage3c(+dq: T16, +dk: T16, +dv: T16, +dr1: T16, +xh1: T16, +rs1: F32, +w: Q) -> R9:
  match w:
    case {qpat('wqT', 'wkT', 'wvT', 'g1')}:
      +dn1 : T16 = t_add(t_add(mv1(dq, wqT), mv1(dk, wkT)), mv1(dv, wvT))
      R9{{dn1, t_add(dr1, ln_bwd(dn1, g1, xh1, rs1))}}
""")
    return "\n".join(o)


# ------------------------------------------------------------------------------------------------
# passes over the per-token record lists
# ------------------------------------------------------------------------------------------------
def passes() -> str:
    o = []
    o.append(f"""def p1a(toks: Toks, +t: U32, +w: Q) -> +List<R1>:
  match toks:
    case TNil{{}}:
      Nil{{}}
    case TCon{{+tk, tt}}:
      Con{{stage1(tk, t, w), p1a(tt, (t + 1 : U32), w)}}

# attention forward: keys / values seen so far, newest first
def p1b(rs: +List<R1>, +t: U32, ks: Tl, vs: Tl) -> +List<R2>:
  match rs:
    case Nil{{}}:
      Nil{{}}
    case {lpat('R1', 'rt', 'q', 'k', 'v')}:
      +ks2 : Tl = TlC{{k, ks}}
      +vs2 : Tl = TlC{{v, vs}}
      Con{{attn_stage(q, ks2, vs2, t), p1b(rt, (t + 1 : U32), ks2, vs2)}}

def p1c(a: +List<R1>, b: +List<R2>, +w: Q) -> +List<R3>:
  match a b:
    case {lpat('R1', 't1', 'e')} {lpat('R2', 't2', 'ctx')}:
      Con{{stage3a1(e, ctx, w), p1c(t1, t2, w)}}
    case _ _:
      Nil{{}}

def p1d(a: +List<R3>, ys: Toks, +w: Q) -> +List<R4>:
  match a ys:
    case {lpat('R3', 't3', 'r1', 'f0', 'f1', 'f2', 'f3')} TCon{{+y, yt}}:
      Con{{stage3a2(r1, f0, f1, f2, f3, y, w), p1d(t3, yt, w)}}
    case _ _:
      Nil{{}}

def p1e(a: +List<R4>, +w: Q) -> +List<R5>:
  match a:
    case Nil{{}}:
      Nil{{}}
    case {lpat('R4', 't4', 'xhf', 'rsf', 'dlog0', 'dlog1')}:
      Con{{stage3b1(dlog0, dlog1, xhf, rsf, w), p1e(t4, w)}}

def p1f(a: +List<R5>, b: +List<R3>, +w: Q) -> +List<R6>:
  match a b:
    case {lpat('R5', 't5', 'dr2')} {lpat('R3', 't3', 'f0', 'f1', 'f2', 'f3', 'xh2', 'rs2')}:
      Con{{stage3b2(dr2, f0, f1, f2, f3, xh2, rs2, w), p1f(t5, t3, w)}}
    case _ _:
      Nil{{}}

# attention backward (token local): needs the same newest-first key / value prefixes
def p1g(a: +List<R1>, b: +List<R2>, c: +List<R6>, +t: U32, ks: Tl, vs: Tl) -> +List<R7>:
  match a b c:
    case {lpat('R1', 't1', 'k', 'v')} {lpat('R2', 't2', 'a')} {lpat('R6', 't6', 'dctx')}:
      +ks2 : Tl = TlC{{k, ks}}
      +vs2 : Tl = TlC{{v, vs}}
      Con{{attn_bwd(a, dctx, ks2, vs2, t), p1g(t1, t2, t6, (t + 1 : U32), ks2, vs2)}}
    case _ _ _:
      Nil{{}}

# key j: dk_j = scale * sum_{{t >= j}} ds_t[j] q_t ,  dv_j = sum_{{t >= j}} a_t[j] dctx_t
# (two separate loops: one loop carrying both accumulators and four lists exceeded the register arity)
def kdk_go(+j: U32, a: +List<R1>, d: +List<R7>, +dk: T16) -> T16:
  match a d:
    case {lpat('R1', 't1', 'q')} {lpat('R7', 't7', 'ds')}:
      kdk_go(j, t1, t7, t_add(dk, t_scale((t_get(ds, j) * fscale() : F32), q)))
    case _ _:
      dk

def kdv_go(+j: U32, b: +List<R2>, c: +List<R6>, +dv: T16) -> T16:
  match b c:
    case {lpat('R2', 't2', 'a')} {lpat('R6', 't6', 'dctx')}:
      kdv_go(j, t2, t6, t_add(dv, t_scale(t_get(a, j), dctx)))
    case _ _:
      dv

# two single-result passes (dk list, dv list), zipped afterwards: building both T16 results in one
# expression exceeded the register arity
def p2k_acc(+a: +List<R1>, +d: +List<R7>, +j: U32, acc: Tl) -> Tl:
  match a d:
    case Con{{_, t1}} Con{{_, t7}}:
      p2k_acc(t1, t7, (j + 1 : U32), TlC{{kdk_go(j, a, d, t_zero()), acc}})
    case _ _:
      acc

def p2v_acc(+b: +List<R2>, +c: +List<R6>, +j: U32, acc: Tl) -> Tl:
  match b c:
    case Con{{_, t2}} Con{{_, t6}}:
      p2v_acc(t2, t6, (j + 1 : U32), TlC{{kdv_go(j, b, c, t_zero()), acc}})
    case _ _:
      acc

def p2k(+a: +List<R1>, +d: +List<R7>, +j: U32) -> Tl:
  tl_rev(p2k_acc(a, d, j, TlN{{}}))

def p2v(+b: +List<R2>, +c: +List<R6>, +j: U32) -> Tl:
  tl_rev(p2v_acc(b, c, j, TlN{{}}))

def zip8(a: Tl, b: Tl) -> +List<R8>:
  match a b:
    case TlC{{x, xt}} TlC{{y, yt}}:
      Con{{R8{{x, y}}, zip8(xt, yt)}}
    case _ _:
      Nil{{}}

def p2(+a: +List<R1>, +b: +List<R2>, +c: +List<R6>, +d: +List<R7>, +j: U32) -> +List<R8>:
  zip8(p2k(a, d, j), p2v(b, c, j))

def p3(a: +List<R1>, b: +List<R7>, c: +List<R8>, d: +List<R6>, +w: Q) -> +List<R9>:
  match a b c d:
    case {lpat('R1', 't1', 'xh1', 'rs1')} {lpat('R7', 't7', 'dq')} {lpat('R8', 't8', 'dk', 'dv')} {lpat('R6', 't6', 'dr1')}:
      Con{{stage3c(dq, dk, dv, dr1, xh1, rs1, w), p3(t1, t7, t8, t6, w)}}
    case _ _ _ _:
      Nil{{}}
""")
    return "\n".join(o)


# ------------------------------------------------------------------------------------------------
# weight / bias / LN gradients: register accumulators looping over the sample's tokens
# ------------------------------------------------------------------------------------------------
LISTS = {"l1": "R1", "l2": "R2", "l3": "R3", "l4": "R4", "l5": "R5", "l6": "R6", "l7": "R7", "l8": "R8", "l9": "R9"}


def _lp(names):
    return ", ".join(f"{n}: +List<{LISTS[n]}>" for n in names)


def _lc(names):
    return ", ".join(names)


def grad_builders() -> str:
    o = []

    def outer(name, srcs, dy, x, extra="", nrows=16):
        """Rank-1 accumulation  dW[o] = sum_t dy_t[o] * x_t, four rows per pass over the tokens.
        The lanes are static (field select, no one-hot dot product) and x_t (e.g. n1 = xh*g + b) is
        formed once per token for all four rows of the pass."""
        lists = [s[0] for s in srcs]
        ex_sig = f"{extra}, " if extra else ""
        ex_call = ", ".join(e.split(":")[0].strip().lstrip("+") for e in extra.split(", ")) + ", " if extra else ""
        pats = " ".join(lpat(LISTS[s[0]], f"t{i}", *s[1:]) for i, s in enumerate(srcs))
        tails = ", ".join(f"t{i}" for i in range(len(srcs)))
        wild = " ".join("_" for _ in srcs)
        shared = _lp(lists).replace(", ", ", +")
        names_ = ", ".join(lists)
        for g in range(nrows // 4):
            lane = lambda i: f"lane{g * 4 + i}(dd)"
            o.append(f"""def gw_{name}_g{g}({ex_sig}{_lp(lists)}, +c0: T16, +c1: T16, +c2: T16, +c3: T16) -> R4t:
  match {' '.join(lists)}:
    case {pats}:
      +xx : T16 = {x}
      +dd : T16 = {dy}
      gw_{name}_g{g}({ex_call}{tails}, t_add(c0, t_scale({lane(0)}, xx)), t_add(c1, t_scale({lane(1)}, xx)), t_add(c2, t_scale({lane(2)}, xx)), t_add(c3, t_scale({lane(3)}, xx)))
    case {wild}:
      R4t{{c0, c1, c2, c3}}

def gw_{name}_p{g}({ex_sig}+{shared}, rows: Mat) -> Mat:
  r4t_rows(gw_{name}_g{g}({ex_call}{names_}, t_zero(), t_zero(), t_zero(), t_zero()), rows)
""")
        chain = "MNil{}"
        for g in reversed(range(nrows // 4)):
            chain = f"gw_{name}_p{g}({ex_call}{names_}, {chain})"
        o.append(f"""def gw_{name}({ex_sig}+{shared}) -> Mat:
  {chain}
""")

    def total(name, srcs, expr, tile=True):
        lists = [s[0] for s in srcs]
        pats = " ".join(lpat(LISTS[s[0]], f"t{i}", *s[1:]) for i, s in enumerate(srcs))
        tails = ", ".join(f"t{i}" for i in range(len(srcs)))
        wild = " ".join("_" for _ in srcs)
        o.append(f"""def sm_{name}_acc({_lp(lists)}, +acc: T16) -> T16:
  match {' '.join(lists)}:
    case {pats}:
      sm_{name}_acc({tails}, t_add(acc, {expr}))
    case {wild}:
      acc

def sm_{name}_t({_lp(lists)}) -> T16:
  sm_{name}_acc({_lc(lists)}, t_zero())

def sm_{name}({_lp(lists)}) -> TV:
  QCon{{sm_{name}_acc({_lc(lists)}, t_zero()), QNil{{}}}}
""")

    n1x = "t_add(t_mul(xh1, g1), b1)"
    outer("q", [("l1", "xh1"), ("l7", "dq")], "dq", n1x, "+g1: T16, +b1: T16")
    outer("k", [("l1", "xh1"), ("l8", "dk")], "dk", n1x, "+g1: T16, +b1: T16")
    outer("v", [("l1", "xh1"), ("l8", "dv")], "dv", n1x, "+g1: T16, +b1: T16")
    outer("o", [("l2", "ctx"), ("l6", "dr1")], "dr1", "ctx")
    n2x = "t_add(t_mul(xh2, g2), b2)"
    for m in range(4):
        outer(f"fc{m}", [("l3", "xh2"), ("l6", f"df{m}")], f"df{m}", n2x, "+g2: T16, +b2: T16")
        outer(f"mp{m}", [("l3", f"f{m}"), ("l5", "dr2")], "dr2", f"t_gelu(f{m})")
    nfx = "t_add(t_mul(xhf, gf), bf)"
    for m in range(2):
        outer(f"te{m}", [("l4", "xhf", f"dlog{m}")], f"dlog{m}", nfx, "+gf: T16, +bf: T16")

    total("bq", [("l7", "dq")], "dq")
    total("bk", [("l8", "dk")], "dk")
    total("bv", [("l8", "dv")], "dv")
    total("bo", [("l6", "dr1")], "dr1")
    for m in range(4):
        total(f"bfc{m}", [("l6", f"df{m}")], f"df{m}")
    total("bmp", [("l5", "dr2")], "dr2")
    total("g1", [("l9", "dn1"), ("l1", "xh1")], "t_mul(dn1, xh1)")
    total("b1", [("l9", "dn1")], "dn1")
    total("g2", [("l6", "dn2"), ("l3", "xh2")], "t_mul(dn2, xh2)")
    total("b2", [("l6", "dn2")], "dn2")
    total("gf", [("l5", "dnf"), ("l4", "xhf")], "t_mul(dnf, xhf)")
    total("bf", [("l5", "dnf")], "dnf")

    o.append("""def de_mat_acc(a: +List<R9>, acc: Mat) -> Mat:
  match a:
    case Nil{}:
      acc
    case Con{R9{_, +de}, t}:
      de_mat_acc(t, MC1{de, acc})

def de_mat(a: +List<R9>) -> Mat:
  mat_rev(de_mat_acc(a, MNil{}))

def loss_sum(a: +List<R4>, +acc: F32) -> F32:
  match a:
    case Nil{}:
      acc
    case Con{R4{_, _, _, _, +l}, t}:
      loss_sum(t, (acc + l : F32))
""")
    return "\n".join(o)


def q_get() -> str:
    return "\n".join(f"def qg_{n}(+w: Q) -> {QT[n]}:\n  match w:\n    case {qpat(n)}:\n      {n}\n" for n in ("g1", "b1", "g2", "b2", "gf", "bf"))


GF_A_PAR = '''def gf_a(+w: Q, +toks: Toks, +l1: +List<R1>, +l2: +List<R2>, +l3: +List<R3>, +l4: +List<R4>, +l5: +List<R5>, +l6: +List<R6>, +l7: +List<R7>, +l8: +List<R8>, +l9: +List<R9>, +demat: Mat) -> G:
  x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 = gf_wte(w, toks, l4, demat) sm_g1(l9, l1) sm_b1(l9) gf_wq(w, l1, l7) sm_bq(l7) gf_wk(w, l1, l8) sm_bk(l8) gf_wv(w, l1, l8) sm_bv(l8) gw_o(l2, l6) sm_bo(l6) sm_g2(l6, l3) sm_b2(l6) gf_wfc(w, l3, l6) gf_bfc(l6) gf_wmp(l3, l5) sm_bmp(l5) sm_gf(l5, l4) sm_bf(l5)
  G{loss_sum(l4, 0.0), x0, demat, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18}

'''
GF_A_SEQ = '''def gf_a(+w: Q, +toks: Toks, +l1: +List<R1>, +l2: +List<R2>, +l3: +List<R3>, +l4: +List<R4>, +l5: +List<R5>, +l6: +List<R6>, +l7: +List<R7>, +l8: +List<R8>, +l9: +List<R9>, +demat: Mat) -> G:
  G{loss_sum(l4, 0.0), gf_wte(w, toks, l4, demat), demat, sm_g1(l9, l1), sm_b1(l9), gf_wq(w, l1, l7), sm_bq(l7), gf_wk(w, l1, l8), sm_bk(l8), gf_wv(w, l1, l8), sm_bv(l8), gw_o(l2, l6), sm_bo(l6), sm_g2(l6, l3), sm_b2(l6), gf_wfc(w, l3, l6), gf_bfc(l6), gf_wmp(l3, l5), sm_bmp(l5), sm_gf(l5, l4), sm_bf(l5)}

'''


def sample4(par: bool = False) -> str:
    return """# gradient fields, each in its own small function (a single G{...} with ~20 argument expressions and
# six live LayerNorm tiles exceeded Bend's 247-register arity)
def gf_wte(+w: Q, +toks: Toks, +l4: +List<R4>, +demat: Mat) -> Mat:
  mat_add(mat_cat(gw_te0(qg_gf(w), qg_bf(w), l4), gw_te1(qg_gf(w), qg_bf(w), l4)), scatter_add(toks, demat, mat_zeros(nV(), 1)))

def gf_wq(+w: Q, +l1: +List<R1>, +l7: +List<R7>) -> Mat:
  gw_q(qg_g1(w), qg_b1(w), l1, l7)

def gf_wk(+w: Q, +l1: +List<R1>, +l8: +List<R8>) -> Mat:
  gw_k(qg_g1(w), qg_b1(w), l1, l8)

def gf_wv(+w: Q, +l1: +List<R1>, +l8: +List<R8>) -> Mat:
  gw_v(qg_g1(w), qg_b1(w), l1, l8)

def gf_wfc(+w: Q, +l3: +List<R3>, +l6: +List<R6>) -> Mat:
  mat_cat(gw_fc0(qg_g2(w), qg_b2(w), l3, l6), mat_cat(gw_fc1(qg_g2(w), qg_b2(w), l3, l6), mat_cat(gw_fc2(qg_g2(w), qg_b2(w), l3, l6), gw_fc3(qg_g2(w), qg_b2(w), l3, l6))))

def gf_bfc(+l6: +List<R6>) -> TV:
  QCon{sm_bfc0_t(l6), QCon{sm_bfc1_t(l6), QCon{sm_bfc2_t(l6), QCon{sm_bfc3_t(l6), QNil{}}}}}

def gf_wmp(+l3: +List<R3>, +l5: +List<R5>) -> Mat:
  mat_zip4(gw_mp0(l3, l5), gw_mp1(l3, l5), gw_mp2(l3, l5), gw_mp3(l3, l5))

""" + (GF_A_PAR if par else GF_A_SEQ) + """def sample_grad4(+w: Q, toks: Toks, tgts: Toks) -> G:
  +toks2 : Toks = toks
  +l1 : +List<R1> = p1a(toks2, 0, w)
  +l2 : +List<R2> = p1b(l1, 0, TlN{}, TlN{})
  +l3 : +List<R3> = p1c(l1, l2, w)
  +l4 : +List<R4> = p1d(l3, tgts, w)
  +l5 : +List<R5> = p1e(l4, w)
  +l6 : +List<R6> = p1f(l5, l3, w)
  +l7 : +List<R7> = p1g(l1, l2, l6, 0, TlN{}, TlN{})
  +l8 : +List<R8> = p2(l1, l2, l6, l7, 0)
  +l9 : +List<R9> = p3(l1, l7, l8, l6, w)
  gf_a(w, toks2, l1, l2, l3, l4, l5, l6, l7, l8, l9, de_mat(l9))
"""
