"""
v3 tile library: matrices whose list CELLS hold the tiles inline.

v2 stored a row as a list of tiles inside a list of rows (MCon -> QCon -> T16: three dependent
hops per row, each a refcounted node). Measured: the same 14x16 @ 16x16^T product takes 55 ms
(4096 lanes x 40 ops) that way and 9 ms when a row is a single T16 inside a single-level list.

Here a Mat is ONE list whose cells carry 1, 2 or 4 tiles directly (a row 16, 32 or 64 wide):
  MC1{a, rest}   MC2{a, b, rest}   MC4{a, b, c, d, rest}
The shapes of this model are fixed (C=16, F=64, V=32 -> widths 1, 4, 2), so every operation is
generated for the three widths from a template. Matrix products compute one OUTPUT TILE COLUMN
at a time as a plain single-level list (Col) and zip the columns into cells at the end.
"""
from tile_gen import tile_ops, IDX

NS = (1, 2, 4)
NAMES = "abcd"


def _tiles(prefix, n):
    return [f"{prefix}{k}" for k in range(n)]


def _nest(con, n, tail):
    return f"{con}{n}"  # unused helper kept for clarity


def mc(n, tiles, rest):
    return f"MC{n}{{{', '.join(tiles)}, {rest}}}"


def types() -> str:
    return """# TV (biases, LN gains): a recursive list of tiles. Recursive functions are never inlined; a
# non-recursive multi-branch width dispatch WAS inlined at every call site and blew the 247-register
# arity of g_add / make_p / sample_grad.
type TV is Data:
  QNil{}
  QCon{h: T16, t: TV}

type Mat is Data:
  MNil{}
  MC1{a: T16, rest: Mat}
  MC2{a: T16, b: T16, rest: Mat}
  MC4{a: T16, b: T16, c: T16, d: T16, rest: Mat}

type Col is Data:
  CNil{}
  CCon{t: T16, rest: Col}
"""


# ----------------------------------------------------------------------------- generic templates
# A GPU lane has only ~360 words of continuation stack. `MC4{.., f(rest)}` keeps ~24 words alive per
# row, so a 16-row width-4 (or 64-row width-1) matrix overflowed it. Every row loop is therefore
# written as a tail-recursive accumulate (reversed) + one mat_rev pass.
def _wrap(name, sig, call, scr_params, scr_names, body):
    """body: the `match` cases text of name_acc; returns both defs."""
    return (f"def {name}_acc({sig}{scr_params}, acc: Mat) -> Mat:\n  match {scr_names}:\n{body}\n\n"
            f"def {name}({sig}{scr_params}) -> Mat:\n  mat_rev({name}_acc({call}{scr_names}, MNil{{}}))\n")


def mat_unary(name, sig, call, tile):
    """Mat -> Mat, same tile expression on every tile. tile(x) -> expr; sig/call are extra params."""
    cases = ["    case MNil{}:\n      acc"]
    for n in NS:
        a = _tiles("a", n)
        cases.append(f"    case {mc(n, a, 'rt')}:\n      {name}_acc({call}rt, {mc(n, [tile(x) for x in a], 'acc')})")
    return _wrap(name, sig, call, "m: Mat", "m", "\n".join(cases))


def mat_binary(name, tile2, sig="", call=""):
    """(Mat, Mat) -> Mat; tile2(x, y) -> expr."""
    cases = []
    for n in NS:
        a, b = _tiles("a", n), _tiles("b", n)
        cases.append(f"    case {mc(n, a, 'at')} {mc(n, b, 'bt')}:\n"
                     f"      {name}_acc({call}at, bt, {mc(n, [tile2(x, y) for x, y in zip(a, b)], 'acc')})")
    cases.append("    case _ _:\n      acc")
    return _wrap(name, sig, call, "a: Mat, b: Mat", "a b", "\n".join(cases))


def tv_cases(n, prefix):
    """pattern for a TV of exactly n tiles"""
    out = "_"
    for k in reversed(range(n)):
        out = f"QCon{{{prefix}{k}, {out}}}"
    return out


def tv_make(n, exprs, tail="QNil{}"):
    out = tail
    for e in reversed(exprs):
        out = f"QCon{{{e}, {out}}}"
    return out


def mat_tv(name, tile2, sig="", call=""):
    """(Mat, TV broadcast) -> Mat: every row combined with the same TV."""
    cases = []
    for n in NS:
        a, b = _tiles("a", n), _tiles("b", n)
        cases.append(f"    case {mc(n, a, 'rt')} {tv_cases(n, 'b')}:\n"
                     f"      {name}_acc({call}rt, b, {mc(n, [tile2(x, y) for x, y in zip(a, b)], 'acc')})")
    cases.append("    case _ _:\n      acc")
    return _wrap(name, sig, call, "m: Mat, +b: TV", "m b", "\n".join(cases))


def mat_rows(name, tile2):
    """(Vec of one F32 per row, Mat) -> Mat; tile2(c, x) with c the row scalar."""
    cases = []
    for n in NS:
        a = _tiles("a", n)
        cases.append(f"    case VCon{{+c, ct}} {mc(n, a, 'rt')}:\n"
                     f"      {name}_acc(ct, rt, {mc(n, [tile2('c', x) for x in a], 'acc')})")
    cases.append("    case _ _:\n      acc")
    return (f"def {name}_acc(sv: Vec, m: Mat, acc: Mat) -> Mat:\n  match sv m:\n" + "\n".join(cases) + "\n\n"
            f"def {name}(sv: Vec, m: Mat) -> Mat:\n  mat_rev({name}_acc(sv, m, MNil{{}}))\n")


def row_reduce(name, expr_fn, binary=False):
    """Mat (or two Mats) -> Vec with one F32 per row; expr_fn(tiles) -> expr."""
    if binary:
        out = [f"def {name}(a: Mat, b: Mat) -> Vec:\n  match a b:"]
        for n in NS:
            x, y = _tiles("a", n), _tiles("b", n)
            out.append(f"    case {mc(n, x, 'at')} {mc(n, y, 'bt')}:\n"
                       f"      VCon{{{expr_fn(list(zip(x, y)))}, {name}(at, bt)}}")
        out.append("    case _ _:\n      VNil{}")
    else:
        out = [f"def {name}(m: Mat) -> Vec:\n  match m:\n    case MNil{{}}:\n      VNil{{}}"]
        for n in NS:
            x = _tiles("a", n)
            out.append(f"    case {mc(n, x, 'rt')}:\n      VCon{{{expr_fn(x)}, {name}(rt)}}")
    return "\n".join(out) + "\n"


def library() -> str:
    o = [types()]
    o.append("""def mat_rev_onto(m: Mat, acc: Mat) -> Mat:
  match m:
    case MNil{}:
      acc
    case MC1{a, rt}:
      mat_rev_onto(rt, MC1{a, acc})
    case MC2{a, b, rt}:
      mat_rev_onto(rt, MC2{a, b, acc})
    case MC4{a, b, c, d, rt}:
      mat_rev_onto(rt, MC4{a, b, c, d, acc})

def mat_rev(m: Mat) -> Mat:
  mat_rev_onto(m, MNil{})

def col_rev_onto(c: Col, acc: Col) -> Col:
  match c:
    case CNil{}:
      acc
    case CCon{x, rt}:
      col_rev_onto(rt, CCon{x, acc})

def col_rev(c: Col) -> Col:
  col_rev_onto(c, CNil{})
""")

    # ---- TV
    o.append("def tv_zeros(+w: Nat) -> TV:\n  match w:\n    case 0n:\n      QNil{}\n    case 1n+p:\n      QCon{t_zero(), tv_zeros(p)}\n")
    o.append("def tv_zero(+n: U32) -> TV:\n  tv_zeros(U32.to_nat(n))\n")
    for nm, op in (("tv_add", "t_add"), ("tv_mul", "t_mul")):
        o.append(f"def {nm}(x: TV, y: TV) -> TV:\n  match x y:\n    case QCon{{xh, xt}} QCon{{yh, yt}}:\n"
                 f"      QCon{{{op}(xh, yh), {nm}(xt, yt)}}\n    case _ _:\n      QNil{{}}\n")

    # ---- elementwise Mat ops
    o.append(mat_binary("mat_add", lambda x, y: f"t_add({x}, {y})"))
    o.append(mat_binary("mat_sub", lambda x, y: f"t_sub({x}, {y})"))
    o.append(mat_binary("mat_mul", lambda x, y: f"t_mul({x}, {y})"))
    o.append(mat_binary("mat_gelu_bwd", lambda f, d: f"t_gelu_bwd({f}, {d})"))
    o.append(mat_unary("mat_scale", "+s: F32, ", "s, ", lambda x: f"t_scale(s, {x})"))
    o.append(mat_unary("mat_gelu", "", "", lambda x: f"t_gelu({x})"))
    o.append(mat_tv("mat_addv", lambda x, y: f"t_add({x}, {y})"))
    o.append(mat_tv("mat_mulv", lambda x, y: f"t_mul({x}, {y})"))
    o.append(mat_rows("mat_addrows", lambda c, x: f"t_addc({c}, {x})"))
    o.append(mat_rows("mat_scalerows", lambda c, x: f"t_scale({c}, {x})"))
    o.append(mat_rows("mat_expm_rows", lambda c, x: f"t_expm({c}, {x})"))

    # LN affine: xhat * g + b
    cases = []
    for n in NS:
        a_, g_, b_ = _tiles("a", n), _tiles("g", n), _tiles("b", n)
        cases.append(f"    case {mc(n, a_, 'rt')} {tv_cases(n, 'g')} {tv_cases(n, 'b')}:\n"
                     f"      mat_affine_acc(rt, g, bb, {mc(n, [f't_add(t_mul({x}, {y}), {z})' for x, y, z in zip(a_, g_, b_)], 'acc')})")
    cases.append("    case _ _ _:\n      acc")
    o.append(_wrap("mat_affine", "", "", "m: Mat, +g: TV, +bb: TV", "m g bb", "\n".join(cases)).replace("mat_affine_acc(m g bb", "mat_affine_acc(m, g, bb").replace("(m g bb, MNil", "(m, g, bb, MNil").replace("m: Mat, +g: TV, +bb: TV, acc: Mat) -> Mat:\n  match m g bb:", "m: Mat, +g: TV, +bb: TV, acc: Mat) -> Mat:\n  match m g bb:"))

    # ---- per-row reductions
    def add_all(terms):
        return "(" + " + ".join(terms) + " : F32)"

    o.append(row_reduce("mat_rowsum", lambda x: add_all([f"t_sum({t})" for t in x])))
    o.append(row_reduce("mat_rowdot", lambda p: add_all([f"t_dot({x}, {y})" for x, y in p]), binary=True))

    def mx(x):
        e = f"t_max({x[0]})"
        for t in x[1:]:
            e = f"F32.max({e}, t_max({t}))"
        return e

    o.append(row_reduce("mat_rowmax", mx))

    # ---- per-row scalar (Vec) helpers
    o.append("""def sv_map_neg(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{F32.neg(h), sv_map_neg(t)}

def sv_map_scale(+s: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(s * h : F32), sv_map_scale(s, t)}

def sv_map_inv(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(1.0 / h : F32), sv_map_inv(t)}

def sv_map_rstd(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(1.0 / F32.sqrt((h + 0.00001 : F32)) : F32), sv_map_rstd(t)}

def sv_sum_acc(v: Vec, +acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      sv_sum_acc(t, (acc + h : F32))
""")

    # ---- column sum of a Mat -> TV
    out = ["def mat_colsum_acc(m: Mat, acc: TV) -> TV:\n  match m acc:"]
    for n in NS:
        a, c = _tiles("a", n), _tiles("c", n)
        out.append(f"    case {mc(n, a, 'rt')} {tv_cases(n, 'c')}:\n"
                   f"      mat_colsum_acc(rt, {tv_make(n, [f't_add({y}, {x})' for x, y in zip(a, c)])})")
    out.append("    case _ _:\n      acc")
    o.append("\n".join(out) + "\n")
    o.append("def mat_colsum(m: Mat, +n: U32) -> TV:\n  mat_colsum_acc(m, tv_zero(n))\n")

    # ---- causal mask: row t keeps lanes <= t (tile k starts at lane 16k)
    cases = ["    case MNil{}:\n      acc"]
    for n in NS:
        a_ = _tiles("a", n)
        cases.append(f"    case {mc(n, a_, 'rt')}:\n      mat_mask_acc(rt, (t + 1 : U32), "
                     + mc(n, [f"t_mask(t, {16 * k}, {x})" for k, x in enumerate(a_)], "acc") + ")")
    o.append("def mat_mask_acc(m: Mat, +t: U32, acc: Mat) -> Mat:\n  match m:\n" + "\n".join(cases)
             + "\n\ndef mat_mask(m: Mat, +t: U32) -> Mat:\n  mat_rev(mat_mask_acc(m, t, MNil{}))\n")

    # ---- cross entropy helpers: p[y] per row, and softmax - onehot
    out = ["def mat_pick(m: Mat, tg: Toks) -> Vec:\n  match m tg:"]
    for n in NS:
        a = _tiles("a", n)
        terms = " + ".join(f"t_dot({x}, t_onehot(U32.sub(y, {16 * k})))" for k, x in enumerate(a))
        out.append(f"    case {mc(n, a, 'rt')} TCon{{+y, yt}}:\n      VCon{{({terms} : F32), mat_pick(rt, yt)}}")
    out.append("    case _ _:\n      VNil{}")
    o.append("\n".join(out) + "\n")
    cases = []
    for n in NS:
        a_ = _tiles("a", n)
        cases.append(f"    case {mc(n, a_, 'rt')} TCon{{+y, yt}}:\n      mat_ce_grad_acc(rt, yt, "
                     + mc(n, [f"t_sub({x}, t_onehot(U32.sub(y, {16 * k})))" for k, x in enumerate(a_)], "acc") + ")")
    cases.append("    case _ _:\n      acc")
    o.append("def mat_ce_grad_acc(m: Mat, tg: Toks, acc: Mat) -> Mat:\n  match m tg:\n" + "\n".join(cases)
             + "\n\ndef mat_ce_grad(m: Mat, tg: Toks) -> Mat:\n  mat_rev(mat_ce_grad_acc(m, tg, MNil{}))\n")
    o.append("""def ce_loss(pv: Vec, +acc: F32) -> F32:
  match pv:
    case VNil{}:
      acc
    case VCon{+p, t}:
      ce_loss(t, (acc - F32.log(F32.max(p, 0.000000000000000000000000000001)) : F32))
""")

    # ---- structure helpers
    o.append("""def mat_zero_cell(+n: U32, rest: Mat) -> Mat:
  match n:
    case 1:
      MC1{t_zero(), rest}
    case 2:
      MC2{t_zero(), t_zero(), rest}
    case _:
      MC4{t_zero(), t_zero(), t_zero(), t_zero(), rest}

def mat_zeros_acc(+r: Nat, +n: U32, acc: Mat) -> Mat:
  match r:
    case 0n:
      acc
    case 1n+p:
      mat_zeros_acc(p, n, mat_zero_cell(n, acc))

def mat_zeros(+r: Nat, +n: U32) -> Mat:
  mat_zeros_acc(r, n, MNil{})

def mat_cat(a: Mat, b: Mat) -> Mat:
  mat_rev_onto(mat_rev(a), b)

def mat_len_acc(m: Mat, +acc: U32) -> U32:
  match m:
    case MNil{}:
      acc
    case MC1{_, rt}:
      mat_len_acc(rt, (acc + 1 : U32))
    case MC2{_, _, rt}:
      mat_len_acc(rt, (acc + 1 : U32))
    case MC4{_, _, _, _, rt}:
      mat_len_acc(rt, (acc + 1 : U32))

def mat_take_acc(m: Mat, +n: U32, acc: Mat) -> Mat:
  match m n:
    case MC1{_, _} 0:
      acc
    case MC2{_, _, _} 0:
      acc
    case MC4{_, _, _, _, _} 0:
      acc
    case MC1{x, rt} _:
      mat_take_acc(rt, U32.sub(n, 1), MC1{x, acc})
    case MC2{x, y, rt} _:
      mat_take_acc(rt, U32.sub(n, 1), MC2{x, y, acc})
    case MC4{x, y, z, w, rt} _:
      mat_take_acc(rt, U32.sub(n, 1), MC4{x, y, z, w, acc})
    case _ _:
      acc

def mat_take(m: Mat, +n: U32) -> Mat:
  mat_rev(mat_take_acc(m, n, MNil{}))

def mat_drop(m: Mat, +n: U32) -> Mat:
  match m n:
    case MC1{_, _} 0:
      m
    case MC2{_, _, _} 0:
      m
    case MC4{_, _, _, _, _} 0:
      m
    case MC1{_, rt} _:
      mat_drop(rt, U32.sub(n, 1))
    case MC2{_, _, rt} _:
      mat_drop(rt, U32.sub(n, 1))
    case MC4{_, _, _, _, rt} _:
      mat_drop(rt, U32.sub(n, 1))
    case _ _:
      m

# pad with zero rows up to a multiple of 16 rows (n = tiles per row)
def mat_pad16(+m: Mat, +n: U32) -> Mat:
  +len : U32 = mat_len_acc(m, 0)
  padn : U32 = U32.mod(U32.sub(16, U32.mod(len, 16)), 16)
  mat_cat(m, mat_zeros(U32.to_nat(padn), n))
""")

    # ---- flat scalars <-> tiles
    o.append("""def tv_tiles(+k: Nat, +v: Vec) -> TV:
  match k:
    case 0n:
      QNil{}
    case 1n+p:
      QCon{t16_of(v), tv_tiles(p, vdrop(16n, v))}

def tv_of(+n: U32, +v: Vec) -> TV:
  tv_tiles(U32.to_nat(n), v)

def mat_of1(+r: Nat, +v: Vec) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MC1{t16_of(v), mat_of1(p, vdrop(16n, v))}

def mat_of2(+r: Nat, +v: Vec) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MC2{t16_of(v), t16_of(vdrop(16n, v)), mat_of2(p, vdrop(32n, v))}

def mat_of4(+r: Nat, +v: Vec) -> Mat:
  match r:
    case 0n:
      MNil{}
    case 1n+p:
      MC4{t16_of(v), t16_of(vdrop(16n, v)), t16_of(vdrop(32n, v)), t16_of(vdrop(48n, v)), mat_of4(p, vdrop(64n, v))}

def mat_of(+r: Nat, +n: U32, +v: Vec) -> Mat:
  match n:
    case 1:
      mat_of1(r, v)
    case 2:
      mat_of2(r, v)
    case _:
      mat_of4(r, v)
""")

    # ---- Col (single-level list of tiles) zips
    o.append("""def zip1_acc(c0: Col, acc: Mat) -> Mat:
  match c0:
    case CCon{x0, t0}:
      zip1_acc(t0, MC1{x0, acc})
    case _:
      acc

def zip1(c0: Col) -> Mat:
  mat_rev(zip1_acc(c0, MNil{}))

def zip2_acc(c0: Col, c1: Col, acc: Mat) -> Mat:
  match c0 c1:
    case CCon{x0, t0} CCon{x1, t1}:
      zip2_acc(t0, t1, MC2{x0, x1, acc})
    case _ _:
      acc

def zip2(c0: Col, c1: Col) -> Mat:
  mat_rev(zip2_acc(c0, c1, MNil{}))

def zip4_acc(c0: Col, c1: Col, c2: Col, c3: Col, acc: Mat) -> Mat:
  match c0 c1 c2 c3:
    case CCon{x0, t0} CCon{x1, t1} CCon{x2, t2} CCon{x3, t3}:
      zip4_acc(t0, t1, t2, t3, MC4{x0, x1, x2, x3, acc})
    case _ _ _ _:
      acc

def zip4(c0: Col, c1: Col, c2: Col, c3: Col) -> Mat:
  mat_rev(zip4_acc(c0, c1, c2, c3, MNil{}))

# Mat of width-1 cells -> Col, and zips of whole Mats (used to assemble transposed weights)
def mat_to_col_acc(m: Mat, acc: Col) -> Col:
  match m:
    case MC1{x, rt}:
      mat_to_col_acc(rt, CCon{x, acc})
    case _:
      acc

def mat_to_col(m: Mat) -> Col:
  col_rev(mat_to_col_acc(m, CNil{}))

def mat_zip2(a: Mat, b: Mat) -> Mat:
  zip2(mat_to_col(a), mat_to_col(b))

def mat_zip4(a: Mat, b: Mat, c: Mat, d: Mat) -> Mat:
  zip4(mat_to_col(a), mat_to_col(b), mat_to_col(c), mat_to_col(d))
""")

    # ---- tiles of the product: 16 consecutive B rows -> one output tile
    def b_nest(n, rows_prefix="b"):
        s = "_"
        for j in reversed(IDX):
            tiles = [f"{rows_prefix}{j}_{k}" for k in range(n)]
            s = mc(n, tiles, s)
        return s

    outs1 = ", ".join(f"t_dot(a0, b{j}_0)" for j in IDX)
    o.append(f"def tile1(+a0: T16, blk: Mat) -> T16:\n  match blk:\n    case {b_nest(1)}:\n      T16{{{outs1}}}\n    case _:\n      t_zero()\n")
    # K = 2 / 4: flat tail-recursive loop over the 16 rows (a function with 16 x K live tiles would
    # exceed Bend's 247-register arity), results pushed on a reversed scalar list, packed at the end
    o.append(f"""def pack16(v: Vec) -> T16:
  match v:
    case {_vnest([f"x{15 - i}" for i in IDX])}:
      T16{{{', '.join(f'x{i}' for i in IDX)}}}
    case _:
      t_zero()
""")
    o.append("""def tile2_go(+a0: T16, +a1: T16, blk: Mat, +cnt: U32, acc: Vec) -> Vec:
  match blk cnt:
    case MC2{_, _, _} 0:
      acc
    case MC2{b0, b1, rt} _:
      tile2_go(a0, a1, rt, U32.sub(cnt, 1), VCon{(t_dot(a0, b0) + t_dot(a1, b1) : F32), acc})
    case _ _:
      acc

def tile2(+a0: T16, +a1: T16, blk: Mat) -> T16:
  pack16(tile2_go(a0, a1, blk, 16, VNil{}))

def tile4_go(+a0: T16, +a1: T16, +a2: T16, +a3: T16, blk: Mat, +cnt: U32, acc: Vec) -> Vec:
  match blk cnt:
    case MC4{_, _, _, _, _} 0:
      acc
    case MC4{b0, b1, b2, b3, rt} _:
      tile4_go(a0, a1, a2, a3, rt, U32.sub(cnt, 1), VCon{((t_dot(a0, b0) + t_dot(a1, b1)) + (t_dot(a2, b2) + t_dot(a3, b3)) : F32), acc})
    case _ _:
      acc

def tile4(+a0: T16, +a1: T16, +a2: T16, +a3: T16, blk: Mat) -> T16:
  pack16(tile4_go(a0, a1, a2, a3, blk, 16, VNil{}))

# one output tile column: the i-th tile of every row of A against one 16-row block of B
def mmcol_acc(a: Mat, +blk: Mat, acc: Col) -> Col:
  match a:
    case MNil{}:
      acc
    case MC1{a0, at}:
      mmcol_acc(at, blk, CCon{tile1(a0, blk), acc})
    case MC2{a0, a1, at}:
      mmcol_acc(at, blk, CCon{tile2(a0, a1, blk), acc})
    case MC4{a0, a1, a2, a3, at}:
      mmcol_acc(at, blk, CCon{tile4(a0, a1, a2, a3, blk), acc})

def mmcol(a: Mat, +blk: Mat) -> Col:
  col_rev(mmcol_acc(a, blk, CNil{}))

# A B^T. B has 16*no rows (no = number of output tiles = 1, 2 or 4)
def mat_nt(+a: Mat, +b: Mat, +no: U32) -> Mat:
  match no:
    case 1:
      zip1(mmcol(a, b))
    case 2:
      zip2(mmcol(a, b), mmcol(a, mat_drop(b, 16)))
    case _:
      zip4(mmcol(a, b), mmcol(a, mat_drop(b, 16)), mmcol(a, mat_drop(b, 32)), mmcol(a, mat_drop(b, 48)))
""")

    # ---- transposes of exactly 16 rows (n tiles wide) -> 16*n rows of width 1
    o.append(blk_t1_def())
    for n in (1, 2):
        a = [[f"x{i}_{k}" for k in range(n)] for i in IDX]
        pat = "_"
        for i in reversed(IDX):
            pat = mc(n, a[i], pat)
        cols = []
        for k in range(n):
            args = ", ".join(a[i][k] for i in IDX)
            cols.append(f"blk_t1({args})")
        body = cols[-1]
        for c in reversed(cols[:-1]):
            body = f"mat_cat({c}, {body})"
        o.append(f"def mat_t{n}(m: Mat) -> Mat:\n  match m:\n    case {pat}:\n      {body}\n    case _:\n      MNil{{}}\n")
    # width-4 input: a 64-variable pattern exceeds Bend's 247-register arity, so peel the four tile
    # columns into width-1 matrices and transpose those
    picks = []
    for k in range(4):
        picks.append(f"    case MC4{{a, b, c, d, rt}} {k}:\n      mat_pick_col_acc(rt, {k}, MC1{{{'abcd'[k]}, acc}})")
    o.append("def mat_pick_col_acc(m: Mat, +k: U32, acc: Mat) -> Mat:\n  match m k:\n" + "\n".join(picks) + "\n    case _ _:\n      acc\n")
    o.append("def mat_pick_col(m: Mat, +k: U32) -> Mat:\n  mat_rev(mat_pick_col_acc(m, k, MNil{}))\n")
    o.append("""def mat_t4(+m: Mat) -> Mat:
  mat_cat(mat_t1(mat_pick_col(m, 0)), mat_cat(mat_t1(mat_pick_col(m, 1)), mat_cat(mat_t1(mat_pick_col(m, 2)), mat_t1(mat_pick_col(m, 3)))))
""")
    o.append("""def mat_t(+m: Mat, +n: U32) -> Mat:
  match n:
    case 1:
      mat_t1(mat_pad16(m, 1))
    case 2:
      mat_t2(mat_pad16(m, 2))
    case _:
      mat_t4(mat_pad16(m, 4))
""")
    o.append(composites())
    return "\n".join(o)


def _vnest(items):
    s = "_"
    for x in reversed(items):
        s = f"VCon{{{x}, {s}}}"
    return s


def blk_t1_def() -> str:
    """16 tiles (rows) -> 16 width-1 cells (columns of the 16x16 block)."""
    cases = "case " + " ".join("T16{" + ", ".join(f"r{j}_{i}" for i in IDX) + "}" for j in IDX) + ":"
    args = ", ".join(f"r{j}: T16" for j in IDX)
    cells = "MNil{}"
    for i in reversed(IDX):
        cells = f"MC1{{T16{{{', '.join(f'r{j}_{i}' for j in IDX)}}}, {cells}}}"
    return f"def blk_t1({args}) -> Mat:\n  match {' '.join(f'r{j}' for j in IDX)}:\n    {cases}\n      {cells}\n"


def adam_ops() -> str:
    o = []
    for nm, sig, call, op in (("adam_m", "+s: F32, +b1: F32, +omb: F32, ", "s, b1, omb", "t_adam_m"),
                              ("adam_v", "+s: F32, +b2: F32, +omb: F32, ", "s, b2, omb", "t_adam_v")):
        # TV
        o.append(f"def tv_{nm}({sig}g: TV, m: TV) -> TV:\n  match g m:\n    case QCon{{gh, gt}} QCon{{mh, mt}}:\n"
                 f"      QCon{{{op}({call}, gh, mh), tv_{nm}({call}, gt, mt)}}\n    case _ _:\n      QNil{{}}\n")
        cases = []
        for n in NS:
            g, m = _tiles("g", n), _tiles("m", n)
            cases.append(f"    case {mc(n, g, 'gt')} {mc(n, m, 'mt')}:\n      mat_{nm}_acc({call}, gt, mt, "
                         + mc(n, [f"{op}({call}, {x}, {y})" for x, y in zip(g, m)], "acc") + ")")
        cases.append("    case _ _:\n      acc")
        o.append(f"def mat_{nm}_acc({sig}g: Mat, m: Mat, acc: Mat) -> Mat:\n  match g m:\n" + "\n".join(cases)
                 + f"\n\ndef mat_{nm}({sig}g: Mat, m: Mat) -> Mat:\n  mat_rev(mat_{nm}_acc({call}, g, m, MNil{{}}))\n")
    # params: p, m, v
    ap = "+lr: F32, +c1: F32, +c2: F32, +eps: F32, "
    o.append(f"def tv_adam_p({ap}p: TV, m: TV, v: TV) -> TV:\n  match p m v:\n    case QCon{{ph, pt}} QCon{{mh, mt}} QCon{{vh, vt}}:\n"
             f"      QCon{{t_adam_p(lr, c1, c2, eps, ph, mh, vh), tv_adam_p(lr, c1, c2, eps, pt, mt, vt)}}\n    case _ _ _:\n      QNil{{}}\n")
    cases = []
    for n in NS:
        p_, m_, v_ = _tiles("p", n), _tiles("m", n), _tiles("v", n)
        cases.append(f"    case {mc(n, p_, 'pt')} {mc(n, m_, 'mt')} {mc(n, v_, 'vt')}:\n      mat_adam_p_acc(lr, c1, c2, eps, pt, mt, vt, "
                     + mc(n, [f"t_adam_p(lr, c1, c2, eps, {x}, {y}, {z})" for x, y, z in zip(p_, m_, v_)], "acc") + ")")
    cases.append("    case _ _ _:\n      acc")
    o.append(f"def mat_adam_p_acc({ap}p: Mat, m: Mat, v: Mat, acc: Mat) -> Mat:\n  match p m v:\n" + "\n".join(cases)
             + f"\n\ndef mat_adam_p({ap}p: Mat, m: Mat, v: Mat) -> Mat:\n  mat_rev(mat_adam_p_acc(lr, c1, c2, eps, p, m, v, MNil{{}}))\n")
    return "\n".join(o)


def composites() -> str:
    return """# --- LayerNorm over rows (xhat / rstd are the backward cache) --------------------------------
def mat_xhat(+m: Mat) -> Mat:
  +mean : Vec = sv_map_scale(inv_c(), mat_rowsum(m))
  +cen : Mat = mat_addrows(sv_map_neg(mean), m)
  var : Vec = sv_map_scale(inv_c(), mat_rowdot(cen, cen))
  mat_scalerows(sv_map_rstd(var), cen)

def mat_rstd(+m: Mat) -> Vec:
  +mean : Vec = sv_map_scale(inv_c(), mat_rowsum(m))
  +cen : Mat = mat_addrows(sv_map_neg(mean), m)
  var : Vec = sv_map_scale(inv_c(), mat_rowdot(cen, cen))
  sv_map_rstd(var)

# dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)),  dxhat = dy * gamma
def mat_ln_bwd(dy: Mat, +xhat: Mat, rstd: Vec, +g: TV) -> Mat:
  +dxh : Mat = mat_mulv(dy, g)
  m1 : Vec = sv_map_scale(inv_c(), mat_rowsum(dxh))
  m2 : Vec = sv_map_scale(inv_c(), mat_rowdot(dxh, xhat))
  mat_scalerows(rstd, mat_sub(mat_addrows(sv_map_neg(m1), dxh), mat_scalerows(m2, xhat)))

# --- softmax over rows, and its backward --------------------------------------------------------
def mat_softmax(+m: Mat) -> Mat:
  mx : Vec = mat_rowmax(m)
  +e : Mat = mat_expm_rows(mx, m)
  s : Vec = mat_rowsum(e)
  mat_scalerows(sv_map_inv(s), e)

# dS = A * (dA - rowsum(A * dA))
def mat_softmax_bwd(+a: Mat, +da: Mat) -> Mat:
  d : Vec = mat_rowdot(a, da)
  mat_mul(a, mat_addrows(sv_map_neg(d), da))

def lin(+x: Mat, +w: Mat, +b: TV, +no: U32) -> Mat:
  mat_addv(mat_nt(x, w, no), b)

# --- Embedding (width 1: C = 16) --------------------------------------------------------------------
def nth0(+m: Mat, +idx: U32) -> T16:
  match m idx:
    case MC1{+x, _} 0:
      x
    case MC1{_, rt} _:
      nth0(rt, U32.sub(idx, 1))
    case _ _:
      t_zero()

def embed(+wte: Mat, toks: Toks, wpe: Mat) -> Mat:
  match toks wpe:
    case TCon{+tk, tt} MC1{p0, pt}:
      MC1{t_add(nth0(wte, tk), p0), embed(wte, tt, pt)}
    case _ _:
      MNil{}

def row_add1(acc: Mat, +idx: U32, d0: T16) -> Mat:
  match acc idx:
    case MC1{x, rt} 0:
      MC1{t_add(x, d0), rt}
    case MC1{x, rt} _:
      MC1{x, row_add1(rt, U32.sub(idx, 1), d0)}
    case _ _:
      acc

def scatter_add(toks: Toks, de: Mat, acc: Mat) -> Mat:
  match toks de:
    case TCon{+tk, tt} MC1{d0, dt}:
      scatter_add(tt, dt, row_add1(acc, tk, d0))
    case _ _:
      acc
"""


def sample3(field_names) -> str:
    """sample_grad for the C=16 / F=64 / V=32 model on width-1/4/2 matrices."""
    P = ", ".join(f"+{n}" for n in field_names + ["wteT", "wqT", "wkT", "wvT", "woT", "wfcT", "wmpT"])
    return f"""def sample_grad(+p: P, toks: Toks, tgts: Toks) -> G:
  match p:
    case P{{{P}}}:
      +toks2 : Toks = toks
      # ---- forward ----
      +e : Mat = embed(wte, toks2, wpe)
      +xh1 : Mat = mat_xhat(e)
      +rs1 : Vec = mat_rstd(e)
      +n1 : Mat = mat_affine(xh1, ln1g, ln1b)
      +q : Mat = lin(n1, wq, bq, 1)
      +k : Mat = lin(n1, wk, bk, 1)
      +v : Mat = lin(n1, wv, bv, 1)
      +kp : Mat = mat_pad16(k, 1)
      +vp : Mat = mat_pad16(v, 1)
      +vT : Mat = mat_t(v, 1)
      +a : Mat = mat_softmax(mat_mask(mat_scale(fscale(), mat_nt(q, kp, 1)), 0))
      +ctx : Mat = mat_nt(a, vT, 1)
      +o : Mat = lin(ctx, wo, bo, 1)
      +r1 : Mat = mat_add(e, o)
      +xh2 : Mat = mat_xhat(r1)
      +rs2 : Vec = mat_rstd(r1)
      +n2 : Mat = mat_affine(xh2, ln2g, ln2b)
      +f : Mat = lin(n2, wfc, bfc, 4)
      +h : Mat = mat_gelu(f)
      +mo : Mat = lin(h, wmp, bmp, 1)
      +r2 : Mat = mat_add(r1, mo)
      +xhf : Mat = mat_xhat(r2)
      +rsf : Vec = mat_rstd(r2)
      +nf : Mat = mat_affine(xhf, lnfg, lnfb)
      +logits : Mat = mat_nt(nf, wte, 2)
      +pr : Mat = mat_softmax(logits)
      +tg : Toks = tgts
      loss : F32 = ce_loss(mat_pick(pr, tg), 0.0)
      # ---- backward ----
      +dlog : Mat = mat_ce_grad(pr, tg)
      dwte_head : Mat = mat_nt(mat_t(dlog, 2), mat_t(nf, 1), 1)
      +dnf : Mat = mat_nt(dlog, wteT, 1)
      dlnfg : TV = mat_colsum(mat_mul(dnf, xhf), 1)
      dlnfb : TV = mat_colsum(dnf, 1)
      +dr2 : Mat = mat_ln_bwd(dnf, xhf, rsf, lnfg)
      dwmp : Mat = mat_nt(mat_t(dr2, 1), mat_t(h, 4), 4)
      dbmp : TV = mat_colsum(dr2, 1)
      +dh : Mat = mat_nt(dr2, wmpT, 4)
      +df : Mat = mat_gelu_bwd(f, dh)
      dwfc : Mat = mat_nt(mat_t(df, 4), mat_t(n2, 1), 1)
      dbfc : TV = mat_colsum(df, 4)
      +dn2 : Mat = mat_nt(df, wfcT, 1)
      dln2g : TV = mat_colsum(mat_mul(dn2, xh2), 1)
      dln2b : TV = mat_colsum(dn2, 1)
      +dr1 : Mat = mat_add(dr2, mat_ln_bwd(dn2, xh2, rs2, ln2g))
      dwo : Mat = mat_nt(mat_t(dr1, 1), mat_t(ctx, 1), 1)
      dbo : TV = mat_colsum(dr1, 1)
      +dctx : Mat = mat_nt(dr1, woT, 1)
      +da : Mat = mat_nt(dctx, vp, 1)
      dv : Mat = mat_take(mat_nt(mat_t(a, 1), mat_t(dctx, 1), 1), nT32())
      +ds : Mat = mat_softmax_bwd(a, da)
      dq : Mat = mat_scale(fscale(), mat_nt(ds, mat_t(k, 1), 1))
      dk : Mat = mat_take(mat_scale(fscale(), mat_nt(mat_t(ds, 1), mat_t(q, 1), 1)), nT32())
      +dq2 : Mat = dq
      +dk2 : Mat = dk
      +dv2 : Mat = dv
      +n1T : Mat = mat_t(n1, 1)
      dwq : Mat = mat_nt(mat_t(dq2, 1), n1T, 1)
      dbq : TV = mat_colsum(dq2, 1)
      dwk : Mat = mat_nt(mat_t(dk2, 1), n1T, 1)
      dbk : TV = mat_colsum(dk2, 1)
      dwv : Mat = mat_nt(mat_t(dv2, 1), n1T, 1)
      dbv : TV = mat_colsum(dv2, 1)
      dn1 : Mat = mat_add(mat_add(mat_nt(dq2, wqT, 1), mat_nt(dk2, wkT, 1)), mat_nt(dv2, wvT, 1))
      +dn1b : Mat = dn1
      dln1g : TV = mat_colsum(mat_mul(dn1b, xh1), 1)
      dln1b : TV = mat_colsum(dn1b, 1)
      +de : Mat = mat_add(dr1, mat_ln_bwd(dn1b, xh1, rs1, ln1g))
      dwpe : Mat = de
      dwte : Mat = mat_add(dwte_head, scatter_add(toks2, de, mat_zeros(nV(), 1)))
      G{{loss, dwte, dwpe, dln1g, dln1b, dwq, dbq, dwk, dbk, dwv, dbv, dwo, dbo, dln2g, dln2b, dwfc, dbfc, dwmp, dbmp, dlnfg, dlnfb}}
"""
