"""
Generates the 16-wide "tile" primitives used by nanogpt/train_tile.bend.

Why tiles: Bend has no dense arrays. A cons list of F32 costs one heap cell and one
dependent load per element, so on a GPU (one load in flight per lane, DRAM latency
~600 ns once the working set leaves L2) a list-based matmul runs 20-60x below what the
same loop does when it stays in cache. A T16 is ONE heap node holding 16 contiguous
F32 fields: one load brings in 16 values and the 16-term dot product is straight-line
code. Vectors are lists of tiles (width = 16 * #tiles); matrices are lists of such rows.
"""

N = 16
IDX = range(N)


def _fields(prefix, plus=True):
    p = "+" if plus else ""
    return ", ".join(f"{p}{prefix}{i}" for i in IDX)


def _tile(exprs):
    return "T16{" + ", ".join(exprs) + "}"


def _un(name, expr_fn, extra=""):
    """T16 -> T16 elementwise: expr_fn(i) uses a{i}."""
    return (f"def {name}({extra}a: T16) -> T16:\n  match a:\n    case T16{{{_fields('a')}}}:\n"
            f"      {_tile(expr_fn(i) for i in IDX)}\n")


def _bin(name, expr_fn):
    return (f"def {name}(a: T16, b: T16) -> T16:\n  match a b:\n"
            f"    case T16{{{_fields('a')}}} T16{{{_fields('b')}}}:\n"
            f"      {_tile(expr_fn(i) for i in IDX)}\n")


def tile_ops(lists: bool = True) -> str:
    """lists=False: only the tile-level primitives (no list-of-tiles types / blk_t)."""
    o = []
    o.append("type Vec is Data:\n  VNil{}\n  VCon{head: F32, tail: Vec}\n")
    o.append("type T16 is Data:\n  T16{" + ", ".join(f"x{i}: F32" for i in IDX) + "}\n")
    if lists:
        o.append("type TV is Data:\n  QNil{}\n  QCon{h: T16, t: TV}\n")
        o.append("type Mat is Data:\n  MNil{}\n  MCon{row: TV, rest: Mat}\n")
    o.append("type Toks is Data:\n  TNil{}\n  TCon{tok: U32, rest: Toks}\n")

    # --- scalar GELU (tanh approximation) and its derivative
    o.append("""def f32_gelu(+x: F32) -> F32:
  +c : F32 = 0.79788456
  +x3 : F32 = (x * x * x : F32)
  inner = (c * (x + 0.044715 * x3 : F32) : F32)
  t = F32.tanh(inner)
  (0.5 * x * (1.0 + t : F32) : F32)

def f32_gelu_grad(+x: F32) -> F32:
  +c : F32 = 0.79788456
  +x2 : F32 = (x * x : F32)
  inner : F32 = (c * (x + 0.044715 * x * x2 : F32) : F32)
  +t : F32 = F32.tanh(inner)
  sech2 : F32 = (1.0 - t * t : F32)
  (0.5 * (1.0 + t : F32) + 0.5 * x * sech2 * c * (1.0 + 0.134145 * x2 : F32) : F32)
""")

    # --- tile elementwise
    o.append(_bin("t_add", lambda i: f"(a{i} + b{i} : F32)"))
    o.append(_bin("t_sub", lambda i: f"(a{i} - b{i} : F32)"))
    o.append(_bin("t_mul", lambda i: f"(a{i} * b{i} : F32)"))
    o.append(_un("t_scale", lambda i: f"(s * a{i} : F32)", "+s: F32, "))
    o.append(_un("t_addc", lambda i: f"(a{i} + s : F32)", "+s: F32, "))
    o.append(_un("t_gelu", lambda i: f"f32_gelu(a{i})"))
    o.append(_bin("t_gelu_bwd", lambda i: f"(b{i} * f32_gelu_grad(a{i}) : F32)"))
    o.append(_un("t_expm", lambda i: f"F32.exp((a{i} - m : F32))", "+m: F32, "))
    o.append(f"def t_copy(a: T16) -> T16:\n  match a:\n    case T16{{{_fields('a')}}}:\n      {_tile(f'a{i}' for i in IDX)}\n")
    # AdamW (wd = 0) on one tile; s = gradient scale (1/(B*T)), omb = 1 - beta
    o.append(f"def t_adam_m(+s: F32, +b1: F32, +omb: F32, g: T16, m: T16) -> T16:\n  match g m:\n"
             f"    case T16{{{_fields('g')}}} T16{{{_fields('m')}}}:\n"
             f"      {_tile(f'(b1 * m{i} + omb * (s * g{i} : F32) : F32)' for i in IDX)}\n")
    o.append(f"def t_adam_v(+s: F32, +b2: F32, +omb: F32, g: T16, v: T16) -> T16:\n  match g v:\n"
             f"    case T16{{{_fields('g')}}} T16{{{_fields('v')}}}:\n"
             f"      {_tile(f'(b2 * v{i} + omb * (s * g{i} : F32) * (s * g{i} : F32) : F32)' for i in IDX)}\n")
    o.append(f"def t_adam_p(+lr: F32, +c1: F32, +c2: F32, +eps: F32, p: T16, m: T16, v: T16) -> T16:\n  match p m v:\n"
             f"    case T16{{{_fields('p')}}} T16{{{_fields('m')}}} T16{{{_fields('v')}}}:\n"
             f"      {_tile(f'(p{i} - lr * (m{i} / c1 : F32) / (F32.sqrt((v{i} / c2 : F32)) + eps : F32) : F32)' for i in IDX)}\n")
    o.append("def t_zero() -> T16:\n  " + _tile("0.0" for _ in IDX) + "\n")
    # causal mask: lane i of a tile starting at `base` is kept iff base + i <= lim
    o.append(
        f"def t_mask(+lim: U32, +base: U32, a: T16) -> T16:\n  match a:\n    case T16{{{_fields('a')}}}:\n"
        f"      +neg_inf : F32 = F32.neg(1000000000.0)\n"
        f"      {_tile(f'Bool.pick(F32, U32.is_le((base + {i} : U32), lim), a{i}, neg_inf)' for i in IDX)}\n")
    o.append(
        f"def t_onehot(+y: U32) -> T16:\n  "
        f"{_tile(f'Bool.pick(F32, U32.is_eq(y, {i}), 1.0, 0.0)' for i in IDX)}\n")

    # --- tile reductions
    def tree_sum(terms):
        return "(" + " + ".join(terms) + " : F32)"

    o.append(f"def t_sum(a: T16) -> F32:\n  match a:\n    case T16{{{_fields('a')}}}:\n      "
             f"{tree_sum(f'a{i}' for i in IDX)}\n")
    o.append(f"def t_dot(a: T16, b: T16) -> F32:\n  match a b:\n"
             f"    case T16{{{_fields('a')}}} T16{{{_fields('b')}}}:\n"
             f"      {tree_sum(f'a{i} * b{i}' for i in IDX)}\n")
    mx = "a0"
    for i in range(1, N):
        mx = f"F32.max({mx}, a{i})"
    o.append(f"def t_max(a: T16) -> F32:\n  match a:\n    case T16{{{_fields('a')}}}:\n      {mx}\n")

    # --- 16x16 block transpose: rows r0..r15 (tiles) -> 16 tiles (column i of the block)
    cases = "case " + " ".join("T16{" + ", ".join(f"r{j}_{i}" for i in IDX) + "}" for j in IDX) + ":"
    cols = "QNil{}"
    for i in reversed(IDX):
        cols = f"QCon{{{_tile(f'r{j}_{i}' for j in IDX)}, {cols}}}"
    args = ", ".join(f"r{j}: T16" for j in IDX)
    if lists:
        o.append(f"def blk_t({args}) -> TV:\n  match {' '.join(f'r{j}' for j in IDX)}:\n    {cases}\n      {cols}\n")

    # --- scalar Vec <-> tile (flat parameters / gradients are plain F32 lists)
    pat = "VNil{}"
    pat = "_"
    nest = pat
    for i in reversed(IDX):
        nest = f"VCon{{a{i}, {nest}}}"
    o.append(f"def t16_of(v: Vec) -> T16:\n  match v:\n    case {nest}:\n      {_tile(f'a{i}' for i in IDX)}\n"
             f"    case _:\n      t_zero()\n")
    nest = "tail"
    for i in reversed(IDX):
        nest = f"VCon{{a{i}, {nest}}}"
    o.append(f"def t16_onto(t: T16, tail: Vec) -> Vec:\n  match t:\n    case T16{{{_fields('a', False)}}}:\n      {nest}\n")
    return "\n".join(o)


def tile_mat_ops() -> str:
    """Matmul / transpose kernels that need 16-row patterns."""
    o = []
    # A B^T for one row of A: every group of 16 rows of B yields one output tile.
    pat = "MNil{}"
    nest = "bt"
    for j in reversed(IDX):
        nest = f"MCon{{b{j}, {nest}}}"
    dots = ", ".join(f"tv_dot(a, b{j}, 0.0)" for j in IDX)
    o.append(f"def row_nt(+a: TV, b: Mat) -> TV:\n  match b:\n    case {nest}:\n"
             f"      QCon{{T16{{{dots}}}, row_nt(a, bt)}}\n    case _:\n      QNil{{}}\n")

    # Contraction-length specialisation: K tiles of A against K tiles of every B row. The K tile
    # dot products are inline straight-line code (no tv_dot loop); one tiny helper per B row keeps
    # each pattern small (Bend caps a constructor/segment at 247 fields).
    for K in (1, 2, 4):
        nest = "_"
        for k in reversed(range(K)):
            nest = f"QCon{{b{k}, {nest}}}"
        params = ", ".join(f"+a{k}: T16" for k in range(K))
        expr = " + ".join(f"t_dot(a{k}, b{k})" for k in range(K))
        o.append(f"def dotk{K}({params}, b: TV) -> F32:\n  match b:\n    case {nest}:\n      ({expr} : F32)\n"
                 f"    case _:\n      0.0\n")
        rows = "bt"
        for j in reversed(IDX):
            rows = f"MCon{{r{j}, {rows}}}"
        call = ", ".join(f"a{k}" for k in range(K))
        outs = ", ".join(f"dotk{K}({call}, r{j})" for j in IDX)
        o.append(f"def row_nt{K}_go({params}, b: Mat) -> TV:\n  match b:\n    case {rows}:\n"
                 f"      QCon{{T16{{{outs}}}, row_nt{K}_go({call}, bt)}}\n    case _:\n      QNil{{}}\n")
        anest = "_"
        for k in reversed(range(K)):
            anest = f"QCon{{a{k}, {anest}}}"
        o.append(f"def row_nt{K}(+a: TV, b: Mat) -> TV:\n  match a:\n    case {anest}:\n"
                 f"      row_nt{K}_go({call}, b)\n    case _:\n      QNil{{}}\n")

    # transpose of a block of 16 rows, w tiles wide -> 16*w rows, one tile long
    rs = ", ".join(f"r{j}: TV" for j in IDX)
    qpat = " ".join(f"QCon{{h{j}, t{j}}}" for j in IDX)
    us = " ".join("_" for _ in IDX)
    hs = ", ".join(f"h{j}" for j in IDX)
    ts = ", ".join(f"t{j}" for j in IDX)
    rn = " ".join(f"r{j}" for j in IDX)
    o.append(f"def tblock(+w: Nat, {rs}) -> Mat:\n  match w {rn}:\n    case 1n+p {qpat}:\n"
             f"      mat_cat(tv_to_rows(blk_t({hs})), tblock(p, {ts}))\n"
             f"    case _ {us}:\n      MNil{{}}\n")

    mpat = "MNil{}"
    nest = "rest"
    for j in reversed(IDX):
        nest = f"MCon{{r{j}, {nest}}}"
    rr = ", ".join(f"r{j}" for j in IDX)
    o.append(f"def mat_t_go(m: Mat, +w: Nat) -> Mat:\n  match m:\n    case {nest}:\n"
             f"      mat_hcat(tblock(w, {rr}), mat_t_go(rest, w))\n    case _:\n      MNil{{}}\n")
    return "\n".join(o)
