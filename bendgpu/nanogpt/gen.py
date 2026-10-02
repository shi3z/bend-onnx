#!/usr/bin/env python3
"""Generate the nanoGPT training step as a Bend program for the bendgpu back end.

One layer, one head, pre-LN, tied embeddings, tanh-GELU, C=16 F=64 V=32 T=14 (the model of nanogpt/bench_train.py).
The step is a pipeline of Arr.map stages over token slots (16 per sample, slots 14 and 15 are padding):

  forward   f1 embed+LN1 -> f2 qkv -> f3 attention -> f4 o-proj+LN2 -> f5 MLP up -> f6 MLP down+LNf+logits+loss
  backward  b1 .. b6 (all per token, reading the tapes of the forward stages)
  gradient  g1 g2 g3: thread (chunk, parameter) sums its parameter's gradient over the tokens of the chunk
  adam      one thread per element of the state array  [params | m | v | step, loss]

  python3 gen.py B > train.bend
"""
import sys, math
import numpy as np

C, T, F, V = 16, 14, 64, 32
O = dict(WTE=0, WPE=512, LN1G=736, LN1B=752, WQ=768, BQ=1024, WK=1040, BK=1296, WV=1312, BV=1568, WO=1584, BO=1840,
         LN2G=1856, LN2B=1872, WFC=1888, BFC=2912, WMP=2976, BMP=4000, LNFG=4016, LNFB=4032)
NPARAM = 4048
LOSS_J = 4048          # parameter slot used by g1 to carry the loss partial sums
LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8


def lit(v):
    return "%.9f" % float(np.float32(v)) if abs(v) > 1e-4 else "%.12f" % float(np.float32(v))


# ------------------------------------------------------------------ tiny emitter DSL
class Fn:
    def __init__(s, name, params, ret):
        s.name, s.params, s.ret, s.lines, s.ind, s.recs, s.lets = name, params, ret, [], 1, [], []

    def raw(s, t):
        s.lines.append("  " * s.ind + t)

    def L(s, t):
        # lets are deferred to the next real line: Bend cannot match a parameter after a let
        for l in s.lets:
            s.raw(l)
        s.lets = []
        s.raw(t)

    def let(s, n, ty, e):
        s.lets.append(f"+{n} : {ty} = {e}")

    def match(s, vs, pats):
        s.raw("match " + " ".join(vs) + ":")
        s.ind += 1
        s.raw("case " + " ".join(pats) + ":")
        s.ind += 1

    def let_rec(s, n, ty, e):
        """a record that is matched later: Bend can only match parameters and fields, so it becomes a parameter of a body def"""
        s.recs.append((n, ty, e))

    def text(s):
        if not s.recs:
            return f"def {s.name}({', '.join(s.params)}) -> {s.ret}:\n" + "\n".join(s.lines) + "\n"
        pn = [p.split(":")[0].strip().lstrip("+") for p in s.params]
        body = f"def {s.name}__b({', '.join([f'+{n}: {t}' for n, t, _ in s.recs] + s.params)}) -> {s.ret}:\n" + "\n".join(s.lines) + "\n"
        wrap = f"def {s.name}({', '.join(s.params)}) -> {s.ret}:\n  {s.name}__b({', '.join([e for _, _, e in s.recs] + pn)})\n"
        return body + wrap


add = lambda a, b: f"({a} + {b} : F32)"
sub = lambda a, b: f"({a} - {b} : F32)"
mul = lambda a, b: f"({a} * {b} : F32)"
div = lambda a, b: f"({a} / {b} : F32)"
pick = lambda ty, c, a, b: f"Bool.pick({ty}, {c}, {a}, {b})"


def chain(terms, init=None):
    acc = init if init is not None else terms[0]
    for t in (terms if init is not None else terms[1:]):
        acc = add(acc, t)
    return acc


def nm(base, k):
    return f"{base}_{k}"


def names(base, n):
    return [nm(base, k) for k in range(n)]


def pat16(base):
    return "V16{" + ", ".join(names(base, 16)) + "}"


def unpackn(f, var, base, n):
    """V16 -> base_0..15 ; V32 -> base_0..31 ; V64 -> base_0..63"""
    if n == 16:
        f.match([var], [pat16(base)])
        return
    parts = n // 16
    tag = {2: "V32", 4: "V64"}[parts]
    sub_names = [f"{base}p{k}" for k in range(parts)]
    f.match([var], [f"{tag}{{{', '.join(sub_names)}}}"])
    f.match(sub_names, ["V16{" + ", ".join(nm(base, 16 * k + c) for c in range(16)) + "}" for k in range(parts)])


def packn(base, n):
    def one(k):
        return "V16{" + ", ".join(nm(base, 16 * k + c) for c in range(16)) + "}"
    if n == 16:
        return one(0)
    parts = n // 16
    tag = {2: "V32", 4: "V64"}[parts]
    return f"{tag}{{" + ", ".join(one(k) for k in range(parts)) + "}"


def zero_vec(n):
    z = "V16{" + ", ".join(["0.0"] * 16) + "}"
    if n == 16:
        return z
    return {2: "V32", 4: "V64"}[n // 16] + "{" + ", ".join([z] * (n // 16)) + "}"


def pp(k):
    return f"pp(s, {k})"


def ix(a, mult, add_):
    return f"({a} * {mult} + {add_} : U32)"


def seltree(vals, k):
    """vals[k] by a tree of picks on the bits of k (len(vals) is a power of two)"""
    n = len(vals)
    if n == 1:
        return vals[0]
    bit = n // 2
    return pick("F32", f"U32.is_ne(U32.and({k}, {bit}), 0)", seltree(vals[bit:], k), seltree(vals[:bit], k))


def matvec(f, out, ins, wbase, bbase, nout, nin, out_ty="F32"):
    """out_j = b_j + sum_i W[wbase + j*nin + i] * ins_i, bound to names out_j"""
    for j in range(nout):
        e = pp(bbase + j) if bbase is not None else "0.0"
        for i in range(nin):
            e = add(e, mul(pp(wbase + j * nin + i), ins[i]))
        f.let(nm(out, j), "F32", e)


def matvec_t(f, out, ins, wbase, nout, nin_rows, stride):
    """out_c = sum_r W[wbase + r*stride + c] * ins_r (transposed use of a weight)"""
    for c in range(nout):
        e = None
        for r in range(nin_rows):
            t = mul(pp(wbase + r * stride + c), ins[r])
            e = t if e is None else add(e, t)
        f.let(nm(out, c), "F32", e)


def layernorm(f, src, xh, n, rs, goff, boff):
    f.let(nm(src, "mean"), "F32", mul(chain(names(src, 16)), "0.0625"))
    for k in range(16):
        f.let(nm(src, f"d{k}"), "F32", sub(nm(src, k), nm(src, "mean")))
    var = chain([mul(nm(src, f"d{k}"), nm(src, f"d{k}")) for k in range(16)])
    f.let(nm(src, "var"), "F32", mul(var, "0.0625"))
    f.let(rs, "F32", div("1.0", f"F32.sqrt({add(nm(src, 'var'), lit(1e-5))})"))
    for k in range(16):
        f.let(nm(xh, k), "F32", mul(nm(src, f"d{k}"), rs))
        f.let(nm(n, k), "F32", add(mul(nm(xh, k), pp(goff + k)), pp(boff + k)))


def layernorm_bwd(f, dn, xh, rs, goff, out, extra=None):
    """out_k = [extra_k +] rs * (dxh_k - mean(dxh) - xh_k * mean(dxh*xh)),  dxh = dn * gamma"""
    for k in range(16):
        f.let(nm(out, f"dx{k}"), "F32", mul(nm(dn, k), pp(goff + k)))
    f.let(nm(out, "m1"), "F32", mul(chain([nm(out, f"dx{k}") for k in range(16)]), "0.0625"))
    f.let(nm(out, "m2"), "F32", mul(chain([mul(nm(out, f"dx{k}"), nm(xh, k)) for k in range(16)]), "0.0625"))
    for k in range(16):
        e = mul(rs, sub(sub(nm(out, f"dx{k}"), nm(out, "m1")), mul(nm(xh, k), nm(out, "m2"))))
        if extra:
            e = add(nm(extra, k), e)
        f.let(nm(out, k), "F32", e)


def gelu(x):
    t = f"F32.tanh({mul(lit(0.79788456), add(x, mul(lit(0.044715), mul(x, mul(x, x)))))})"
    return None, t


# ------------------------------------------------------------------ program
def gen(B):
    N = 16 * B
    DN = int(math.log2(N))
    assert 1 << DN == N
    K = max(1, min(256, N // 1024))
    L = N // K
    KD = int(math.log2(K))
    GD = KD + 12
    inv_bt = np.float32(1.0) / (np.float32(B) * np.float32(T))

    out = []
    out.append("import Base\nimport ../prelude/arr.bend as A\n")
    out.append("# generated by bendgpu/nanogpt/gen.py -- do not edit\n")
    out.append("type V16 is Data:\n  V16{" + ", ".join(f"v{k}: F32" for k in range(16)) + "}\n")
    out.append("type V32 is Data:\n  V32{w0: V16, w1: V16}\n")
    out.append("type V64 is Data:\n  V64{x0: V16, x1: V16, x2: V16, x3: V16}\n")
    recs = {
        "R1": [("e", 16), ("xh1", 16), ("n1", 16), ("rs1", 0)],
        "QKV": [("q", 16), ("k", 16), ("v", 16)],
        "AT": [("a", 16), ("ctx", 16)],
        "R2": [("r1", 16), ("xh2", 16), ("n2", 16), ("rs2", 0)],
        "MU": [("f", 64), ("h", 64)],
        "R3": [("xhf", 16), ("nf", 16), ("rsf", 0), ("dl", 32), ("loss", 0)],
        "BK1": [("dnf", 16), ("dr2", 16)],
        "DF": [("df", 64)],
        "BK3": [("dn2", 16), ("dr1", 16)],
        "BK4": [("dctx", 16), ("ds", 16)],
        "BK5": [("dq", 16), ("dk", 16), ("dv", 16)],
        "BK6": [("dn1", 16), ("de", 16)],
    }
    vt = {0: "F32", 16: "V16", 32: "V32", 64: "V64"}
    for r, fs in recs.items():
        out.append(f"type {r} is Data:\n  {r}{{" + ", ".join(f"{r.lower()}_{n}: {vt[k]}" for n, k in fs) + "}\n")

    def arrT(t):
        return f"A.Arr<&2, {t}>"

    out.append(f"def pp(+s: {arrT('F32')}, +k: U32) -> F32:\n  A.Arr.get(F32, 14n, s, k, 0.0)\n")
    for r, fs in recs.items():
        z = r + "{" + ", ".join(("0.0" if k == 0 else zero_vec(k)) for _, k in fs) + "}"
        out.append(f"def z_{r}() -> {r}:\n  {z}\n")
        out.append(f"def ld_{r}(+a: {arrT(r)}, +k: U32) -> {r}:\n  A.Arr.get({r}, {DN}n, a, k, z_{r}())\n")

    def unrec(f, var, r, pre=""):
        fs = recs[r]
        f.match([var], [r + "{" + ", ".join(f"{pre}{n}" for n, _ in fs) + "}"])

    # element accessors  R_field(r, k)
    for r, fs in recs.items():
        for n, k in fs:
            if k == 0:
                continue
            f = Fn(f"{r}_{n}", [f"+r: {r}", "+k: U32"], "F32")
            unrec(f, "r", r, "z")
            if k == 16:
                unpackn(f, f"z{n}", "u", 16)
            else:
                unpackn(f, f"z{n}", "u", k)
            f.L(seltree(names("u", k), "k"))
            out.append(f.text())

    # fields as vectors:  load record, then unpack the requested fields to scalar names
    def load(f, var, r, arr, idx, fields):
        """+var : r = ld_r(arr, idx); bind scalar names <var>_<field>_<k> for the fields listed"""
        f.let_rec(var, r, f"ld_{r}({arr}, {idx})")
        unrec(f, var, r, var + "_")
        for n in fields:
            k = dict(recs[r])[n]
            if k:
                unpackn(f, f"{var}_{n}", f"{var}{n}", k)

    def vn(var, field, k):
        return f"{var}{field}_{k}"

    def hdr(f):
        f.let("t", "U32", "U32.mod(i, 16)")
        f.let("ph", "U32", "U32.mod(U32.div(i, 16), 4)")
        f.let("base", "U32", "(i - t : U32)")

    # ---- f1: embedding + LN1
    f = Fn("f1", ["+i: U32", f"+s: {arrT('F32')}", f"+tk: {arrT('U32')}"], "R1")
    hdr(f)
    f.let("x", "U32", "A.Arr.get(U32, 7n, tk, (ph * 16 + t : U32), 0)")
    for c in range(16):
        f.let(nm("e", c), "F32", add(pp(f"({'x'} * 16 + {c} : U32)"), pp(f"(t * 16 + {O['WPE'] + c} : U32)")))
    layernorm(f, "e", "xh", "n", "rs", O["LN1G"], O["LN1B"])
    f.L(f"R1{{{packn('e', 16)}, {packn('xh', 16)}, {packn('n', 16)}, rs}}")
    out.append(f.text())

    # ---- f2: q k v
    f = Fn("f2", ["+i: U32", f"+s: {arrT('F32')}", f"+r1s: {arrT('R1')}"], "QKV")
    load(f, "a", "R1", "r1s", "i", ["n1"])
    ins = [vn("a", "n1", k) for k in range(16)]
    matvec(f, "q", ins, O["WQ"], O["BQ"], 16, 16)
    matvec(f, "k", ins, O["WK"], O["BK"], 16, 16)
    matvec(f, "v", ins, O["WV"], O["BV"], 16, 16)
    f.L(f"QKV{{{packn('q', 16)}, {packn('k', 16)}, {packn('v', 16)}}}")
    out.append(f.text())

    # per-key helpers of the attention
    f = Fn("att_sc", ["+j: U32", "+base: U32", f"+qkvs: {arrT('QKV')}", "+q: V16"], "F32")
    f.let_rec("kr", "QKV", "ld_QKV(qkvs, (base + j : U32))")
    unrec(f, "kr", "QKV", "kz")
    f.match(["kzk", "q"], [pat16("k"), pat16("q")])
    f.L(mul(chain([mul(nm("q", c), nm("k", c)) for c in range(16)]), "0.25"))
    out.append(f.text())

    f = Fn("att_ctx", ["+j: U32", "+base: U32", f"+qkvs: {arrT('QKV')}", "+w: F32", "+acc: V16"], "V16")
    f.let_rec("kr", "QKV", "ld_QKV(qkvs, (base + j : U32))")
    unrec(f, "kr", "QKV", "kz")
    f.match(["kzv", "acc"], [pat16("v"), pat16("a")])
    f.L("V16{" + ", ".join(add(nm("a", c), mul("w", nm("v", c))) for c in range(16)) + "}")
    out.append(f.text())

    # ---- f3: attention
    f = Fn("f3", ["+i: U32", f"+qkvs: {arrT('QKV')}"], "AT")
    f.let("t", "U32", "U32.mod(i, 16)")
    f.let("base", "U32", "(i - t : U32)")
    f.let_rec("me", "QKV", "ld_QKV(qkvs, i)")
    unrec(f, "me", "QKV", "m")
    f.let("qv", "V16", "mq")
    NEG = "(0.0 - 1000000000.0 : F32)"
    for j in range(16):
        f.let(f"sc{j}", "F32", pick("F32", f"U32.is_le({j}, t)", f"att_sc({j}, base, qkvs, qv)", NEG))
    mx = "sc0"
    for j in range(1, 16):
        f.let(f"mx{j}", "F32", pick("F32", f"F32.is_gt(sc{j}, {mx})", f"sc{j}", mx))
        mx = f"mx{j}"
    for j in range(16):
        f.let(f"ex{j}", "F32", f"F32.exp({sub(f'sc{j}', mx)})")
    f.let("sm", "F32", chain([f"ex{j}" for j in range(16)]))
    f.let("inv", "F32", div("1.0", "sm"))
    for j in range(16):
        f.let(nm("pa", j), "F32", mul(f"ex{j}", "inv"))
    acc = zero_vec(16)
    for j in range(16):
        f.let(f"c{j}", "V16", pick("V16", f"U32.is_le({j}, t)", f"att_ctx({j}, base, qkvs, {nm('pa', j)}, {acc})", acc))
        acc = f"c{j}"
    f.L(f"AT{{{packn('pa', 16)}, c15}}")
    out.append(f.text())

    # ---- f4: o-proj, residual, LN2
    f = Fn("f4", ["+i: U32", f"+s: {arrT('F32')}", f"+r1s: {arrT('R1')}", f"+ats: {arrT('AT')}"], "R2")
    load(f, "a", "R1", "r1s", "i", ["e"])
    load(f, "b", "AT", "ats", "i", ["ctx"])
    matvec(f, "o", [vn("b", "ctx", k) for k in range(16)], O["WO"], O["BO"], 16, 16)
    for c in range(16):
        f.let(nm("r", c), "F32", add(vn("a", "e", c), nm("o", c)))
    layernorm(f, "r", "xh", "n", "rs", O["LN2G"], O["LN2B"])
    f.L(f"R2{{{packn('r', 16)}, {packn('xh', 16)}, {packn('n', 16)}, rs}}")
    out.append(f.text())

    # ---- f5: MLP up
    f = Fn("f5", ["+i: U32", f"+s: {arrT('F32')}", f"+r2s: {arrT('R2')}"], "MU")
    load(f, "a", "R2", "r2s", "i", ["n2"])
    matvec(f, "f", [vn("a", "n2", k) for k in range(16)], O["WFC"], O["BFC"], F, 16)
    for m in range(F):
        x = nm("f", m)
        f.let(nm("h", m), "F32",
              mul(mul("0.5", x), add("1.0", f"F32.tanh({mul(lit(0.79788456), add(x, mul(lit(0.044715), mul(x, mul(x, x)))))})")))
    f.L(f"MU{{{packn('f', 64)}, {packn('h', 64)}}}")
    out.append(f.text())

    # ---- f6: MLP down, residual, LNf, logits, softmax, loss, dlogits
    f = Fn("f6", ["+i: U32", f"+s: {arrT('F32')}", f"+tk: {arrT('U32')}", f"+r2s: {arrT('R2')}", f"+mus: {arrT('MU')}"], "R3")
    hdr(f)
    f.let("y", "U32", "A.Arr.get(U32, 7n, tk, (ph * 16 + t + 64 : U32), 0)")
    load(f, "a", "R2", "r2s", "i", ["r1"])
    load(f, "b", "MU", "mus", "i", ["h"])
    matvec(f, "mo", [vn("b", "h", k) for k in range(F)], O["WMP"], O["BMP"], 16, F)
    for c in range(16):
        f.let(nm("r", c), "F32", add(vn("a", "r1", c), nm("mo", c)))
    layernorm(f, "r", "xh", "n", "rs", O["LNFG"], O["LNFB"])
    matvec(f, "lg", names("n", 16), O["WTE"], None, V, 16)
    mx = "lg_0"
    for v in range(1, V):
        f.let(f"mx{v}", "F32", pick("F32", f"F32.is_gt({nm('lg', v)}, {mx})", nm("lg", v), mx))
        mx = f"mx{v}"
    for v in range(V):
        f.let(nm("ex", v), "F32", f"F32.exp({sub(nm('lg', v), mx)})")
    f.let("sm", "F32", chain(names("ex", V)))
    f.let("inv", "F32", div("1.0", "sm"))
    f.let("valid", "Bool", "U32.is_lt(t, 14)")
    f.let("ey", "F32", seltree(names("ex", V), "y"))
    f.let("py", "F32", mul("ey", "inv"))
    f.let("loss", "F32", pick("F32", "valid", f"(0.0 - F32.log({pick('F32', 'F32.is_gt(py, 0.000000000000000000000000000001)', 'py', '0.000000000000000000000000000001')}) : F32)", "0.0"))
    for v in range(V):
        f.let(nm("dl", v), "F32", pick("F32", "valid", sub(mul(nm("ex", v), "inv"), pick("F32", f"U32.is_eq(y, {v})", "1.0", "0.0")), "0.0"))
    f.L(f"R3{{{packn('xh', 16)}, {packn('n', 16)}, rs, {packn('dl', 32)}, loss}}")
    out.append(f.text())

    # ---- b1: d nf, LNf backward -> dr2
    f = Fn("b1", ["+i: U32", f"+s: {arrT('F32')}", f"+r3s: {arrT('R3')}"], "BK1")
    load(f, "a", "R3", "r3s", "i", ["xhf", "dl"])
    f.let("rsf", "F32", "a_rsf")
    matvec_t(f, "dnf", [vn("a", "dl", v) for v in range(V)], O["WTE"], 16, V, 16)
    layernorm_bwd(f, "dnf", "axhf", "rsf", O["LNFG"], "dr")
    f.L(f"BK1{{{packn('dnf', 16)}, {packn('dr', 16)}}}")
    out.append(f.text())

    # ---- b2: df = Wmp^T dr2 * gelu'(f)
    f = Fn("b2", ["+i: U32", f"+s: {arrT('F32')}", f"+b1s: {arrT('BK1')}", f"+mus: {arrT('MU')}"], "DF")
    load(f, "a", "BK1", "b1s", "i", ["dr2"])
    load(f, "b", "MU", "mus", "i", ["f"])
    matvec_t(f, "g", [vn("a", "dr2", c) for c in range(16)], O["WMP"], F, 16, F)
    for m in range(F):
        x = vn("b", "f", m)
        f.let(nm("th", m), "F32", f"F32.tanh({mul(lit(0.79788456), add(x, mul(lit(0.044715), mul(x, mul(x, x)))))})")
        gp = add(mul("0.5", add("1.0", nm("th", m))),
                 mul(mul(mul("0.5", x), sub("1.0", mul(nm("th", m), nm("th", m)))),
                     mul(lit(0.79788456), add("1.0", mul(lit(0.134145), mul(x, x))))))
        f.let(nm("df", m), "F32", mul(nm("g", m), gp))
    f.L(f"DF{{{packn('df', 64)}}}")
    out.append(f.text())

    # ---- b3: dn2 = Wfc^T df ; LN2 backward -> dr1 = dr2 + ...
    f = Fn("b3", ["+i: U32", f"+s: {arrT('F32')}", f"+dfs: {arrT('DF')}", f"+r2s: {arrT('R2')}", f"+b1s: {arrT('BK1')}"], "BK3")
    load(f, "a", "DF", "dfs", "i", ["df"])
    load(f, "b", "R2", "r2s", "i", ["xh2"])
    load(f, "c", "BK1", "b1s", "i", ["dr2"])
    f.let("rs2", "F32", "b_rs2")
    matvec_t(f, "dn", [vn("a", "df", m) for m in range(F)], O["WFC"], 16, F, 16)
    for k in range(16):
        f.let(nm("cdr", k), "F32", vn("c", "dr2", k))
    layernorm_bwd(f, "dn", "bxh2", "rs2", O["LN2G"], "dr1", extra="cdr")
    f.L(f"BK3{{{packn('dn', 16)}, {packn('dr1', 16)}}}")
    out.append(f.text())

    # ---- b4: dctx, attention score gradient
    f = Fn("att_da", ["+j: U32", "+base: U32", f"+qkvs: {arrT('QKV')}", "+d: V16"], "F32")
    f.let_rec("kr", "QKV", "ld_QKV(qkvs, (base + j : U32))")
    unrec(f, "kr", "QKV", "kz")
    f.match(["kzv", "d"], [pat16("v"), pat16("d")])
    f.L(chain([mul(nm("d", c), nm("v", c)) for c in range(16)]))
    out.append(f.text())

    f = Fn("b4", ["+i: U32", f"+s: {arrT('F32')}", f"+b3s: {arrT('BK3')}", f"+qkvs: {arrT('QKV')}", f"+ats: {arrT('AT')}"], "BK4")
    f.let("t", "U32", "U32.mod(i, 16)")
    f.let("base", "U32", "(i - t : U32)")
    load(f, "a", "BK3", "b3s", "i", ["dr1"])
    load(f, "b", "AT", "ats", "i", ["a"])
    matvec_t(f, "dc", [vn("a", "dr1", o) for o in range(16)], O["WO"], 16, 16, 16)
    f.let("dcv", "V16", packn("dc", 16))
    for j in range(16):
        f.let(nm("da", j), "F32", pick("F32", f"U32.is_le({j}, t)", f"att_da({j}, base, qkvs, dcv)", "0.0"))
    f.let("dot", "F32", chain([mul(vn("b", "a", j), nm("da", j)) for j in range(16)]))
    for j in range(16):
        f.let(nm("ds", j), "F32", mul(vn("b", "a", j), sub(nm("da", j), "dot")))
    f.L(f"BK4{{{packn('dc', 16)}, {packn('ds', 16)}}}")
    out.append(f.text())

    # ---- b5: dq, dk, dv
    f = Fn("att_kacc", ["+j: U32", "+base: U32", f"+qkvs: {arrT('QKV')}", "+w: F32", "+acc: V16"], "V16")
    f.let_rec("kr", "QKV", "ld_QKV(qkvs, (base + j : U32))")
    unrec(f, "kr", "QKV", "kz")
    f.match(["kzk", "acc"], [pat16("k"), pat16("a")])
    f.L("V16{" + ", ".join(add(nm("a", c), mul("w", nm("k", c))) for c in range(16)) + "}")
    out.append(f.text())

    # dk/dv accumulate over later tokens tt >= t of the sample: dk += ds[tt][t] * q[tt], dv += A[tt][t] * dctx[tt]
    f = Fn("att_kv", ["+tt: U32", "+t: U32", "+base: U32", f"+qkvs: {arrT('QKV')}", f"+ats: {arrT('AT')}", f"+b4s: {arrT('BK4')}", "+acc: V32"], "V32")
    f.let_rec("qr", "QKV", "ld_QKV(qkvs, (base + tt : U32))")
    f.let_rec("br", "BK4", "ld_BK4(b4s, (base + tt : U32))")
    unrec(f, "qr", "QKV", "qz")
    f.match(["qzq"], [pat16("q")])
    f.let("wa", "F32", "AT_a(ld_AT(ats, (base + tt : U32)), t)")
    f.let("wd", "F32", "BK4_ds(ld_BK4(b4s, (base + tt : U32)), t)")
    unrec(f, "br", "BK4", "bz")
    f.match(["bzdctx", "acc"], [pat16("c"), "V32{w0, w1}"])
    f.match(["w0", "w1"], [pat16("k"), pat16("v")])
    f.L("V32{V16{" + ", ".join(add(nm("k", c), mul("wd", nm("q", c))) for c in range(16)) + "}, V16{" +
        ", ".join(add(nm("v", c), mul("wa", nm("c", c))) for c in range(16)) + "}}")
    out.append(f.text())

    f = Fn("b5", ["+i: U32", f"+qkvs: {arrT('QKV')}", f"+ats: {arrT('AT')}", f"+b4s: {arrT('BK4')}"], "BK5")
    f.let("t", "U32", "U32.mod(i, 16)")
    f.let("base", "U32", "(i - t : U32)")
    load(f, "b", "BK4", "b4s", "i", ["ds"])
    acc = zero_vec(16)
    for j in range(16):
        f.let(f"kq{j}", "V16", pick("V16", f"U32.is_le({j}, t)", f"att_kacc({j}, base, qkvs, {vn('b', 'ds', j)}, {acc})", acc))
        acc = f"kq{j}"
    acc = zero_vec(32)
    for tt in range(16):
        f.let(f"kv{tt}", "V32", pick("V32", f"U32.is_ge({tt}, t)", f"att_kv({tt}, t, base, qkvs, ats, b4s, {acc})", acc))
        acc = f"kv{tt}"
    f.L("b5_fin(kv15, kq15)")
    out.append(f.text())
    g = Fn("b5_fin", ["+kv: V32", "+kq: V16"], "BK5")
    g.match(["kv"], ["V32{w0, w1}"])
    g.match(["w0", "w1"], [pat16("k"), pat16("v")])
    g.match(["kq"], [pat16("q")])
    g.L("BK5{V16{" + ", ".join(mul(nm("q", c), "0.25") for c in range(16)) + "}, V16{" +
        ", ".join(mul(nm("k", c), "0.25") for c in range(16)) + "}, " + packn("v", 16) + "}")
    out.insert(len(out) - 1, g.text())

    # ---- b6: dn1 and LN1 backward -> de
    f = Fn("b6", ["+i: U32", f"+s: {arrT('F32')}", f"+b5s: {arrT('BK5')}", f"+b3s: {arrT('BK3')}", f"+r1s: {arrT('R1')}"], "BK6")
    load(f, "a", "BK5", "b5s", "i", ["dq", "dk", "dv"])
    load(f, "b", "BK3", "b3s", "i", ["dr1"])
    load(f, "c", "R1", "r1s", "i", ["xh1"])
    f.let("rs1", "F32", "c_rs1")
    for k in range(16):
        e = None
        for o_ in range(16):
            for w, d in (("WQ", "dq"), ("WK", "dk"), ("WV", "dv")):
                t_ = mul(pp(O[w] + o_ * 16 + k), vn("a", d, o_))
                e = t_ if e is None else add(e, t_)
        f.let(nm("dn", k), "F32", e)
    for k in range(16):
        f.let(nm("cdr", k), "F32", vn("b", "dr1", k))
    layernorm_bwd(f, "dn", "cxh1", "rs1", O["LN1G"], "de", extra="cdr")
    f.L(f"BK6{{{packn('dn', 16)}, {packn('de', 16)}}}")
    out.append(f.text())

    # ================================================================= gradients
    def gstage(name, tapes, groups):
        """tapes: [(param, array type text)]; groups: ascending [(hi, term_body_fn | None)] covering j from 0.
        Thread i is chunk c = i/4096, parameter j = i%4096; it sums the parameter's gradient term over the tokens of
        the chunk (a gap or a parameter beyond the last group has gradient 0)."""
        tp = [f"+{p}: {t}" for p, t in tapes]
        ta = [p for p, _ in tapes]
        e = "0.0"
        for gi in reversed(range(len(groups))):
            hi, body = groups[gi]
            if body is None:
                e = pick("F32", f"U32.is_lt(j, {hi})", "0.0", e)
                continue
            tf = Fn(f"{name}_t{gi}", ["+tk: U32", "+j: U32"] + tp, "F32")
            body(tf)
            out.append(tf.text())
            lf = Fn(f"{name}_l{gi}", ["+n: Nat", "+tk: U32", "+j: U32"] + tp + ["+acc: F32"], "F32")
            lf.L("match n:")
            lf.L("  case 0n:")
            lf.L("    acc")
            lf.L("  case 1n+p:")
            lf.L(f"    {name}_l{gi}(p, (tk + 1 : U32), j, {', '.join(ta)}, {add('acc', f'{name}_t{gi}(tk, j, ' + ', '.join(ta) + ')')})")
            out.append(lf.text())
            e = pick("F32", f"U32.is_lt(j, {hi})", f"{name}_l{gi}({L}n, tk0, j, {', '.join(ta)}, 0.0)", e)
        f = Fn(name, ["+i: U32"] + tp, "F32")
        f.let("j", "U32", "U32.mod(i, 4096)")
        f.let("c", "U32", "U32.div(i, 4096)")
        f.let("tk0", "U32", f"(c * {L} : U32)")
        f.L(e)
        out.append(f.text())

    def div16(off):
        return f"U32.div(U32.sub(j, {off}), 16)", f"U32.mod(U32.sub(j, {off}), 16)"

    def term_of(spec):
        """spec = list of (var, rec, arr) loads and the product expression builder"""
        def mk(loads, expr):
            def body(tf):
                for var, rec, arr in loads:
                    tf.let(var, rec, f"ld_{rec}({arr}, tk)")
                tf.L(expr)
            return body
        return mk
    T_ = term_of(None)

    def outer(off, arec_a, va, fa, arr_a, vb, fb, arr_b, rec_b):
        # sum_tk A_field[row] * B_field[col], row = k/16, col = k%16
        r_, c_ = div16(off)
        return T_([(va, arec_a, arr_a), (vb, rec_b, arr_b)], mul(f"{arec_a}_{fa}({va}, {r_})", f"{rec_b}_{fb}({vb}, {c_})"))

    def vec(off, rec, fld, arr, other=None):
        c = f"U32.mod(U32.sub(j, {off}), 16)"
        if other is None:
            return T_([("a", rec, arr)], f"{rec}_{fld}(a, {c})")
        orec, ofld, oarr = other
        return T_([("a", rec, arr), ("b", orec, oarr)], mul(f"{rec}_{fld}(a, {c})", f"{orec}_{ofld}(b, {c})"))

    # g1: wte head | wmp | bmp | lnfg | lnfb | loss
    gstage("g1", [("r3s", arrT("R3")), ("b1s", arrT("BK1")), ("mus", arrT("MU"))], [
        (512, T_([("a", "R3", "r3s")], mul("R3_dl(a, U32.div(j, 16))", "R3_nf(a, U32.mod(j, 16))"))),
        (O["WMP"], None),
        (O["BMP"], T_([("a", "BK1", "b1s"), ("m", "MU", "mus")],
                      mul("BK1_dr2(a, U32.div(U32.sub(j, %d), 64))" % O["WMP"], "MU_h(m, U32.mod(U32.sub(j, %d), 64))" % O["WMP"]))),
        (O["LNFG"], vec(O["BMP"], "BK1", "dr2", "b1s")),
        (O["LNFB"], vec(O["LNFG"], "BK1", "dnf", "b1s", ("R3", "xhf", "r3s"))),
        (NPARAM, vec(O["LNFB"], "BK1", "dnf", "b1s")),
        (NPARAM + 1, lambda tf: (tf.let_rec("a", "R3", "ld_R3(r3s, tk)"), tf.match(["a"], ["R3{z1, z2, z3, z4, zl}"]), tf.L("zl"))),
    ])
    # g2: wo | bo | ln2g | ln2b | wfc | bfc
    gstage("g2", [("ats", arrT("AT")), ("r2s", arrT("R2")), ("dfs", arrT("DF")), ("b3s", arrT("BK3"))], [
        (O["WO"], None),
        (O["BO"], outer(O["WO"], "BK3", "a", "dr1", "b3s", "b", "ctx", "ats", "AT")),
        (O["LN2G"], vec(O["BO"], "BK3", "dr1", "b3s")),
        (O["LN2B"], vec(O["LN2G"], "BK3", "dn2", "b3s", ("R2", "xh2", "r2s"))),
        (O["WFC"], vec(O["LN2B"], "BK3", "dn2", "b3s")),
        (O["BFC"], outer(O["WFC"], "DF", "a", "df", "dfs", "b", "n2", "r2s", "R2")),
        (O["WMP"], T_([("a", "DF", "dfs")], f"DF_df(a, U32.sub(j, {O['BFC']}))")),
    ])
    # g3: wte embed | wpe | ln1g | ln1b | wq | bq | wk | bk | wv | bv
    def term_embed(tf):
        tf.let("t", "U32", "U32.mod(tk, 16)")
        tf.let("ph", "U32", "U32.mod(U32.div(tk, 16), 4)")
        tf.let("x", "U32", "A.Arr.get(U32, 7n, tks, (ph * 16 + t : U32), 0)")
        tf.let("a", "BK6", "ld_BK6(b6s, tk)")
        tf.L(pick("F32", "U32.is_eq(x, U32.div(j, 16))", "BK6_de(a, U32.mod(j, 16))", "0.0"))

    def term_wpe(tf):
        tf.let("a", "BK6", "ld_BK6(b6s, tk)")
        tf.L(pick("F32", f"U32.is_eq(U32.mod(tk, 16), U32.div(U32.sub(j, {O['WPE']}), 16))",
                  f"BK6_de(a, U32.mod(U32.sub(j, {O['WPE']}), 16))", "0.0"))

    def qkv_out(off, fld):
        return outer(off, "BK5", "a", fld, "b5s", "b", "n1", "r1s", "R1")

    gstage("g3", [("r1s", arrT("R1")), ("b5s", arrT("BK5")), ("b6s", arrT("BK6")), ("tks", arrT("U32"))], [
        (512, term_embed),
        (O["LN1G"], term_wpe),
        (O["LN1B"], vec(O["LN1G"], "BK6", "dn1", "b6s", ("R1", "xh1", "r1s"))),
        (O["WQ"], vec(O["LN1B"], "BK6", "dn1", "b6s")),
        (O["BQ"], qkv_out(O["WQ"], "dq")),
        (O["WK"], vec(O["BQ"], "BK5", "dq", "b5s")),
        (O["BK"], qkv_out(O["WK"], "dk")),
        (O["WV"], vec(O["BK"], "BK5", "dk", "b5s")),
        (O["BV"], qkv_out(O["WV"], "dv")),
        (O["WO"], vec(O["BV"], "BK5", "dv", "b5s")),
    ])

    # ================================================================= Adam on the state [p | m | v | step loss]
    ga = f"g1s, g2s, g3s"
    f = Fn("gsum", ["+n: Nat", "+c: U32", "+j: U32", f"+g1s: {arrT('F32')}", f"+g2s: {arrT('F32')}", f"+g3s: {arrT('F32')}", "+acc: F32"], "F32")
    f.L("match n:")
    f.L("  case 0n:")
    f.L("    acc")
    f.L("  case 1n+p:")
    rd = lambda g: f"A.Arr.get(F32, {GD}n, {g}, (c * 4096 + j : U32), 0.0)"
    f.L(f"    gsum(p, (c + 1 : U32), j, {ga}, {add(add(add('acc', rd('g1s')), rd('g2s')), rd('g3s'))})")
    out.append(f.text())

    f = Fn("lsum", ["+n: Nat", "+c: U32", f"+g1s: {arrT('F32')}", "+acc: F32"], "F32")
    f.L("match n:")
    f.L("  case 0n:")
    f.L("    acc")
    f.L("  case 1n+p:")
    f.L(f"    lsum(p, (c + 1 : U32), g1s, {add('acc', f'A.Arr.get(F32, {GD}n, g1s, (c * 4096 + {NPARAM} : U32), 0.0)')})")
    out.append(f.text())

    f = Fn("upd", ["+sel: U32", "+j: U32", f"+s: {arrT('F32')}", f"+g1s: {arrT('F32')}", f"+g2s: {arrT('F32')}", f"+g3s: {arrT('F32')}"], "F32")
    f.let("g", "F32", mul(f"gsum({K}n, 0, j, {ga}, 0.0)", lit(inv_bt)))
    f.let("st", "F32", add("pp(s, 12288)", "1.0"))
    f.let("mo", "F32", pp("(4096 + j : U32)"))
    f.let("vo", "F32", pp("(8192 + j : U32)"))
    f.let("po", "F32", pp("j"))
    f.let("mm", "F32", add(mul(lit(B1), "mo"), mul(lit(1 - B1), "g")))
    f.let("vv", "F32", add(mul(lit(B2), "vo"), mul(lit(1 - B2), mul("g", "g"))))
    f.let("c1", "F32", sub("1.0", f"F32.pow({lit(B1)}, st)"))
    f.let("c2", "F32", sub("1.0", f"F32.pow({lit(B2)}, st)"))
    f.let("pn", "F32", sub("po", div(mul(lit(LR), div("mm", "c1")), add("F32.sqrt(" + div("vv", "c2") + ")", lit(EPS)))))
    f.L(pick("F32", "U32.is_eq(sel, 0)", "pn", pick("F32", "U32.is_eq(sel, 1)", "mm", "vv")))
    out.append(f.text())

    f = Fn("adam", ["+i: U32", f"+s: {arrT('F32')}", f"+g1s: {arrT('F32')}", f"+g2s: {arrT('F32')}", f"+g3s: {arrT('F32')}"], "F32")
    f.let("j", "U32", "U32.mod(i, 4096)")
    f.let("sel", "U32", "U32.div(i, 4096)")
    keep = pp("i")
    param = pick("F32", f"U32.is_lt(j, {NPARAM})", f"upd(sel, j, s, {ga})", keep)
    misc = pick("F32", "U32.is_eq(i, 12288)", add("pp(s, 12288)", "1.0"),
                pick("F32", "U32.is_eq(i, 12289)", mul(f"lsum({K}n, 0, g1s, 0.0)", lit(inv_bt)), "0.0"))
    f.L(pick("F32", "U32.is_lt(i, 12288)", param, misc))
    out.append(f.text())

    # ---- the step
    stages = [
        ("r1s", "R1", "f1", "F32, U32", "s, tk", DN), ("qkvs", "QKV", "f2", "F32, R1", "s, r1s", DN),
        ("ats", "AT", "f3", "QKV", "qkvs", DN), ("r2s", "R2", "f4", "F32, R1, AT", "s, r1s, ats", DN),
        ("mus", "MU", "f5", "F32, R2", "s, r2s", DN), ("r3s", "R3", "f6", "F32, U32, R2, MU", "s, tk, r2s, mus", DN),
        ("b1s", "BK1", "b1", "F32, R3", "s, r3s", DN), ("dfs", "DF", "b2", "F32, BK1, MU", "s, b1s, mus", DN),
        ("b3s", "BK3", "b3", "F32, DF, R2, BK1", "s, dfs, r2s, b1s", DN),
        ("b4s", "BK4", "b4", "F32, BK3, QKV, AT", "s, b3s, qkvs, ats", DN),
        ("b5s", "BK5", "b5", "QKV, AT, BK4", "qkvs, ats, b4s", DN),
        ("b6s", "BK6", "b6", "F32, BK5, BK3, R1", "s, b5s, b3s, r1s", DN),
        ("g1s", "F32", "g1", "R3, BK1, MU", "r3s, b1s, mus", GD), ("g2s", "F32", "g2", "AT, R2, DF, BK3", "ats, r2s, dfs, b3s", GD),
        ("g3s", "F32", "g3", "R1, BK5, BK6, U32", "r1s, b5s, b6s, tk", GD),
    ]
    out.append("def s_depth() -> Nat:\n  14n\n")
    out.append("def s_init(j: U32) -> F32:\n  0.0\n")
    out.append("def tk_depth() -> Nat:\n  7n\n")
    out.append("def tk_init(+j: U32) -> U32:\n  " + tk_chain() + "\n")
    lines = [f"def train(+s: {arrT('F32')}, +tk: {arrT('U32')}) -> {arrT('F32')}:"]
    for var, ty, fn, ins, args, d in stages:
        types = [t.strip() for t in ins.split(",")]
        n = len(types)
        targs = ", ".join(f"~{t}" for t in types) + f", ~{ty}"
        lines.append(f"  +{var} : {arrT(ty)} = A.Arr.map{n}({targs}, ~{fn}, {d}n, {args}, 0)")
    lines.append(f"  A.Arr.map4(~F32, ~F32, ~F32, ~F32, ~F32, ~adam, 14n, s, g1s, g2s, g3s, 0)")
    out.append("\n".join(lines) + "\n")
    return out


def tk_chain():
    chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!.:01"
    phrases = ["BEND IS FAST!  ", "BEND ON GPU!   ", "BEND RUNS AI!  ", "BEND PARALLEL! "]
    vals = [0] * 128
    for p, ph in enumerate(phrases):
        ids = [chars.index(c) for c in ph]
        for t in range(T):
            vals[p * 16 + t] = ids[t]
            vals[64 + p * 16 + t] = ids[t + 1]
    e = "0"
    for j in reversed(range(128)):
        if vals[j]:
            e = f"Bool.pick(U32, U32.is_eq(j, {j}), {vals[j]}, {e})"
    return e


if __name__ == "__main__":
    print("\n".join(gen(int(sys.argv[1]))))
