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
    K = max(1, min(1024, N // 64))
    L = N // K
    KD = int(math.log2(K))
    GD = KD + 8
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
    # Thread (job r, chunk c) owns the 16 consecutive parameters 16r..16r+15 and sums their gradient over the tokens
    # of chunk c. Consecutive threads share the job (same code path, static field indices); each loads only the
    # fields it needs. The result is a V16 per (job, chunk); Adam adds the chunks up.
    NJOB = 256
    z16 = zero_vec(16)
    out.append(f"def z16() -> V16:\n  {z16}\n")
    f = Fn("v16sel", ["+v: V16", "+k: U32"], "F32")
    f.match(["v"], [pat16("u")])
    f.L(seltree(names("u", 16), "k"))
    out.append(f.text())

    def mkjob(stage, tapes, r, recs, needs, upd, step=1, off=0):
        """recs: [(var, rec, arr)] loaded at token tk; needs: [(var, field, elem)] in record-field order, elem = None for
        the 16 lanes of a V16 field, an int e for the scalar e of a V16/V32/V64 field, ('blk', b) for lane block b;
        upd(env) -> 16 update expressions. Returns the name of the loop function."""
        tp = [f"+{p}: {t}" for p, t in tapes]
        ta = [p for p, _ in tapes]
        nm_ = f"{stage}_j{r}"
        sf = Fn(nm_ + "s", ["+tk: U32"] + tp + ["+acc: V16"], "V16")
        for var, rec, arr in recs:
            sf.let_rec(var, rec, f"ld_{rec}({arr}, tk)")
        env = {}
        done = set()
        for var, fld, elem in needs:
            rec = dict((v, rc) for v, rc, _ in recs)[var]
            if var not in done:
                unrec(sf, var, rec, var + "_")
                done.add(var)
            k = dict(recs_all[rec])[fld]
            base = f"{var}{fld}"
            fv = f"{var}_{fld}"
            if k == 16:
                blk, lane = 0, elem
                sf.match([fv], [pat16(base)])
            else:
                tag = {32: "V32", 64: "V64"}[k]
                if isinstance(elem, tuple):
                    blk, lane = elem[1], None
                else:
                    blk, lane = elem // 16, elem % 16
                hn = [f"{base}h{q}" for q in range(k // 16)]
                sf.match([fv], [f"{tag}{{{', '.join(hn)}}}"])
                sf.match([hn[blk]], [pat16(base)])
            if isinstance(elem, int) or elem is None and k == 16 and False:
                env[(var, fld)] = nm(base, lane)
            else:
                env[(var, fld)] = [nm(base, c) for c in range(16)]
        sf.match(["acc"], [pat16("ac")])
        ups = upd(env)
        sf.L("V16{" + ", ".join(add(nm("ac", c), ups[c]) for c in range(16)) + "}")
        out.append(sf.text())
        lf = Fn(nm_ + "l", ["+n: Nat", "+tk: U32"] + tp + ["+acc: V16"], "V16")
        lf.L("match n:")
        lf.L("  case 0n:")
        lf.L("    acc")
        lf.L("  case 1n+p:")
        lf.L(f"    {nm_}l(p, (tk + {step} : U32), {', '.join(ta)}, {nm_}s(tk, {', '.join(ta)}, acc))")
        out.append(lf.text())
        return nm_ + "l", step, off

    recs_all = recs

    def dispatch(stage, tapes, leaves):
        """leaves: {r: (loopfn, step)}; thread i = (r, c) with r = i / K"""
        tp = [f"+{p}: {t}" for p, t in tapes]
        ta = [p for p, _ in tapes]

        def tree(lo, hi):
            if hi - lo == 1:
                if lo not in leaves:
                    return "z16()"
                lf_, st, off = leaves[lo]
                n_ = L // st
                return f"{lf_}({n_}n, {'tk0' if off == 0 else f'(tk0 + {off} : U32)'}, {', '.join(ta)}, z16())"
            mid = (lo + hi) // 2
            return pick("V16", f"U32.is_lt(r, {mid})", tree(lo, mid), tree(mid, hi))
        f = Fn(stage, ["+i: U32"] + tp, "V16")
        f.let("r", "U32", f"U32.div(i, {K})")
        f.let("c", "U32", f"U32.mod(i, {K})")
        f.let("tk0", "U32", f"(c * {L} : U32)")
        f.L(tree(0, NJOB))
        out.append(f.text())

    # ---- G1: R3 (xhf nf dl) BK1 (dnf dr2) MU (h)
    t1 = [("r3s", arrT("R3")), ("b1s", arrT("BK1")), ("mus", arrT("MU"))]
    leaves = {}
    for v in range(V):   # wte head: dWte[v][c] += dl_v * nf_c
        leaves[v] = mkjob("g1", t1, v, [("a", "R3", "r3s")], [("a", "nf", None), ("a", "dl", v)],
                          lambda e: [mul(e[("a", "dl")], x) for x in e[("a", "nf")]])
    for o_ in range(16):  # wmp[o][16 blocks of 64]
        for blk in range(4):
            r = (O["WMP"] + o_ * 64 + blk * 16) // 16
            leaves[r] = mkjob("g1", t1, r, [("a", "BK1", "b1s"), ("m", "MU", "mus")],
                              [("a", "dr2", o_), ("m", "h", ("blk", blk))],
                              lambda e: [mul(e[("a", "dr2")], x) for x in e[("m", "h")]])
    leaves[O["BMP"] // 16] = mkjob("g1", t1, O["BMP"] // 16, [("a", "BK1", "b1s")], [("a", "dr2", None)], lambda e: list(e[("a", "dr2")]))
    leaves[O["LNFG"] // 16] = mkjob("g1", t1, O["LNFG"] // 16, [("a", "BK1", "b1s"), ("b", "R3", "r3s")],
                                    [("a", "dnf", None), ("b", "xhf", None)],
                                    lambda e: [mul(x, y) for x, y in zip(e[("a", "dnf")], e[("b", "xhf")])])
    leaves[O["LNFB"] // 16] = mkjob("g1", t1, O["LNFB"] // 16, [("a", "BK1", "b1s")], [("a", "dnf", None)], lambda e: list(e[("a", "dnf")]))
    # loss: its own job (needs the scalar field `loss`)
    def mkloss():
        nm_ = "g1_jloss"
        sf = Fn(nm_ + "s", ["+tk: U32"] + [f"+{p}: {t}" for p, t in t1] + ["+acc: V16"], "V16")
        sf.let_rec("a", "R3", "ld_R3(r3s, tk)")
        unrec(sf, "a", "R3", "a_")
        sf.match(["acc"], [pat16("ac")])
        sf.L("V16{" + ", ".join([add("ac_0", "a_loss")] + [nm("ac", c) for c in range(1, 16)]) + "}")
        out.append(sf.text())
        ta = ", ".join(p for p, _ in t1)
        lf = Fn(nm_ + "l", ["+n: Nat", "+tk: U32"] + [f"+{p}: {t}" for p, t in t1] + ["+acc: V16"], "V16")
        lf.L("match n:"); lf.L("  case 0n:"); lf.L("    acc"); lf.L("  case 1n+p:")
        lf.L(f"    {nm_}l(p, (tk + 1 : U32), {ta}, {nm_}s(tk, {ta}, acc))")
        out.append(lf.text())
        return nm_ + "l", 1, 0
    leaves[NPARAM // 16] = mkloss()
    dispatch("g1", t1, leaves)

    # ---- G2: AT (ctx) R2 (xh2 n2) DF (df) BK3 (dn2 dr1)
    t2 = [("ats", arrT("AT")), ("r2s", arrT("R2")), ("dfs", arrT("DF")), ("b3s", arrT("BK3"))]
    leaves = {}
    for o_ in range(16):
        r = (O["WO"] + o_ * 16) // 16
        leaves[r] = mkjob("g2", t2, r, [("a", "BK3", "b3s"), ("b", "AT", "ats")], [("a", "dr1", o_), ("b", "ctx", None)],
                          lambda e: [mul(e[("a", "dr1")], x) for x in e[("b", "ctx")]])
    r = O["BO"] // 16
    leaves[r] = mkjob("g2", t2, r, [("a", "BK3", "b3s")], [("a", "dr1", None)], lambda e: list(e[("a", "dr1")]))
    r = O["LN2G"] // 16
    leaves[r] = mkjob("g2", t2, r, [("a", "BK3", "b3s"), ("b", "R2", "r2s")], [("a", "dn2", None), ("b", "xh2", None)],
                      lambda e: [mul(x, y) for x, y in zip(e[("a", "dn2")], e[("b", "xh2")])])
    r = O["LN2B"] // 16
    leaves[r] = mkjob("g2", t2, r, [("a", "BK3", "b3s")], [("a", "dn2", None)], lambda e: list(e[("a", "dn2")]))
    for m_ in range(F):
        r = (O["WFC"] + m_ * 16) // 16
        leaves[r] = mkjob("g2", t2, r, [("a", "DF", "dfs"), ("b", "R2", "r2s")], [("a", "df", m_), ("b", "n2", None)],
                          lambda e: [mul(e[("a", "df")], x) for x in e[("b", "n2")]])
    for blk in range(4):
        r = (O["BFC"] + blk * 16) // 16
        leaves[r] = mkjob("g2", t2, r, [("a", "DF", "dfs")], [("a", "df", ("blk", blk))], lambda e: list(e[("a", "df")]))
    dispatch("g2", t2, leaves)

    # ---- G3: R1 (xh1 n1) BK5 (dq dk dv) BK6 (dn1 de) + tokens
    t3 = [("r1s", arrT("R1")), ("b5s", arrT("BK5")), ("b6s", arrT("BK6")), ("tks", arrT("U32"))]
    leaves = {}

    def mkembed(v):   # wte[v] += sum over tokens with x == v of de
        nm_ = f"g3_je{v}"
        ptp = [f"+{p}: {t}" for p, t in t3]
        ta = ", ".join(p for p, _ in t3)
        sf = Fn(nm_ + "s", ["+tk: U32"] + ptp + ["+acc: V16"], "V16")
        sf.let_rec("a", "BK6", "ld_BK6(b6s, tk)")
        unrec(sf, "a", "BK6", "a_")
        sf.match(["a_de"], [pat16("de")])
        sf.match(["acc"], [pat16("ac")])
        sf.let("x", "U32", "A.Arr.get(U32, 7n, tks, ((U32.mod(U32.div(tk, 16), 4)) * 16 + U32.mod(tk, 16) : U32), 0)")
        sf.L("V16{" + ", ".join(add(nm("ac", c), pick("F32", f"U32.is_eq(x, {v})", nm("de", c), "0.0")) for c in range(16)) + "}")
        out.append(sf.text())
        lf = Fn(nm_ + "l", ["+n: Nat", "+tk: U32"] + ptp + ["+acc: V16"], "V16")
        lf.L("match n:"); lf.L("  case 0n:"); lf.L("    acc"); lf.L("  case 1n+p:")
        lf.L(f"    {nm_}l(p, (tk + 1 : U32), {ta}, {nm_}s(tk, {ta}, acc))")
        out.append(lf.text())
        return nm_ + "l", 1, 0
    for v in range(V):
        leaves[v] = mkembed(v)
    for t_ in range(T):   # wpe[t] += de of token t of every sample: stride 16
        r = (O["WPE"] + t_ * 16) // 16
        leaves[r] = mkjob("g3", t3, r, [("a", "BK6", "b6s")], [("a", "de", None)], lambda e: list(e[("a", "de")]), step=16, off=t_)
    # the wpe job starts at token tk0 + t: encode the start through the dispatch
    r = O["LN1G"] // 16
    leaves[r] = mkjob("g3", t3, r, [("a", "R1", "r1s"), ("b", "BK6", "b6s")], [("a", "xh1", None), ("b", "dn1", None)],
                      lambda e: [mul(x, y) for x, y in zip(e[("a", "xh1")], e[("b", "dn1")])])
    r = O["LN1B"] // 16
    leaves[r] = mkjob("g3", t3, r, [("a", "BK6", "b6s")], [("a", "dn1", None)], lambda e: list(e[("a", "dn1")]))
    for (woff, boff, fld) in ((O["WQ"], O["BQ"], "dq"), (O["WK"], O["BK"], "dk"), (O["WV"], O["BV"], "dv")):
        for o_ in range(16):
            r = (woff + o_ * 16) // 16
            leaves[r] = mkjob("g3", t3, r, [("a", "BK5", "b5s"), ("b", "R1", "r1s")], [("a", fld, o_), ("b", "n1", None)],
                              lambda e, fld=fld: [mul(e[("a", fld)], x) for x in e[("b", "n1")]])
        r = boff // 16
        leaves[r] = mkjob("g3", t3, r, [("a", "BK5", "b5s")], [("a", fld, None)], lambda e, fld=fld: list(e[("a", fld)]))
    dispatch("g3", t3, leaves)

    # ================================================================= reduce chunks, Adam
    Q = min(32, K)
    QD = int(math.log2(Q))
    GD2 = 8 + QD
    per = K // Q
    # gred: thread (job r, group q) adds the gradient chunks q*per .. (q+1)*per-1 of the three gradient stages
    gtp = [f"+g1s: {arrT('V16')}", f"+g2s: {arrT('V16')}", f"+g3s: {arrT('V16')}"]
    f = Fn("gred_s", ["+x: V16", "+y: V16", "+z: V16", "+acc: V16"], "V16")
    f.match(["x"], [pat16("x")])
    f.match(["y"], [pat16("y")])
    f.match(["z"], [pat16("z")])
    f.match(["acc"], [pat16("a")])
    f.L("V16{" + ", ".join(add(add(add(nm("a", c), nm("x", c)), nm("y", c)), nm("z", c)) for c in range(16)) + "}")
    out.append(f.text())
    f = Fn("gred_l", ["+n: Nat", "+k: U32"] + gtp + ["+acc: V16"], "V16")
    f.L("match n:")
    f.L("  case 0n:")
    f.L("    acc")
    f.L("  case 1n+p:")
    ld = lambda g: f"A.Arr.get(V16, {GD}n, {g}, k, z16())"
    f.L(f"    gred_l(p, (k + 1 : U32), g1s, g2s, g3s, gred_s({ld('g1s')}, {ld('g2s')}, {ld('g3s')}, acc))")
    out.append(f.text())
    f = Fn("gred", ["+i: U32"] + gtp, "V16")
    f.let("r", "U32", f"U32.div(i, {Q})")
    f.let("q", "U32", f"U32.mod(i, {Q})")
    f.L(f"gred_l({per}n, (r * {K} + q * {per} : U32), g1s, g2s, g3s, z16())")
    out.append(f.text())

    ga = "grs"
    f = Fn("gsum", ["+n: Nat", "+c: U32", "+r: U32", "+col: U32", f"+grs: {arrT('V16')}", "+acc: F32"], "F32")
    f.L("match n:")
    f.L("  case 0n:")
    f.L("    acc")
    f.L("  case 1n+p:")
    f.L(f"    gsum(p, (c + 1 : U32), r, col, grs, {add('acc', f'v16sel(A.Arr.get(V16, {GD2}n, grs, (r * {Q} + c : U32), z16()), col)')})")
    out.append(f.text())

    f = Fn("upd", ["+sel: U32", "+j: U32", f"+s: {arrT('F32')}", f"+grs: {arrT('V16')}"], "F32")
    f.let("g", "F32", mul(f"gsum({Q}n, 0, U32.div(j, 16), U32.mod(j, 16), grs, 0.0)", lit(inv_bt)))
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

    f = Fn("adam", ["+i: U32", f"+s: {arrT('F32')}", f"+grs: {arrT('V16')}"], "F32")
    f.let("j", "U32", "U32.mod(i, 4096)")
    f.let("sel", "U32", "U32.div(i, 4096)")
    keep = pp("i")
    param = pick("F32", f"U32.is_lt(j, {NPARAM})", "upd(sel, j, s, grs)", keep)
    misc = pick("F32", "U32.is_eq(i, 12288)", add("pp(s, 12288)", "1.0"),
                pick("F32", "U32.is_eq(i, 12289)", mul(f"gsum({Q}n, 0, {NPARAM // 16}, 0, grs, 0.0)", lit(inv_bt)), "0.0"))
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
        ("g1s", "V16", "g1", "R3, BK1, MU", "r3s, b1s, mus", GD), ("g2s", "V16", "g2", "AT, R2, DF, BK3", "ats, r2s, dfs, b3s", GD),
        ("g3s", "V16", "g3", "R1, BK5, BK6, U32", "r1s, b5s, b6s, tk", GD),
        ("grs", "V16", "gred", "V16, V16, V16", "g1s, g2s, g3s", GD2),
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
    lines.append(f"  A.Arr.map2(~F32, ~V16, ~F32, ~adam, 14n, s, grs, 0)")
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
