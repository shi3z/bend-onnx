#!/usr/bin/env python3
"""
Train nanoGPT in pure Bend (hand-written backward + AdamW) and compare against
PyTorch (CPU / CUDA) on the same model, same init, same data, same optimizer.

  python3 nanogpt/bench_train.py verify            # loss curves: Bend vs PyTorch
  python3 nanogpt/bench_train.py bench             # throughput sweep
"""
import os, sys, re, math, time, json, subprocess, argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from nanogpt.model import GPT, GPTConfig

BEND = "/home/shi3z/snap/antigravity-cli/common/.bend/bin/bend"
CLANG_BIN = os.environ.get("BEND_CLANG_BIN", "")
WORK = os.environ.get("BEND_WORK", "/tmp/bend_train")

CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!.:01"
PHRASES = ["BEND IS FAST!  ", "BEND ON GPU!   ", "BEND RUNS AI!  ", "BEND PARALLEL! "]
V = len(CHARS)
enc = lambda s: [CHARS.index(c) for c in s]
X = [enc(p)[:-1] for p in PHRASES]
Y = [enc(p)[1:] for p in PHRASES]
T = len(X[0])

LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8


# ------------------------------------------------------------------ model / params
def make_model(C, seed=42):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=16, vocab_size=V, n_layer=1, n_head=1, n_embd=C, dropout=0.0, bias=True)
    m = GPT(cfg)
    m.transformer.h[0].mlp.gelu = nn.GELU(approximate="tanh")  # Bend implements tanh-GELU
    return m


def flat_params(m, C):
    """Flatten in the order the Bend record P is unflattened (wpe truncated to T rows)."""
    g = lambda t: t.detach().cpu().numpy().astype(np.float32).ravel()
    b = m.transformer.h[0]
    w = b.attn.c_attn.weight.detach().cpu().numpy()
    bb = b.attn.c_attn.bias.detach().cpu().numpy()
    parts = [g(m.transformer.wte.weight), g(m.transformer.wpe.weight[:T]),
             g(b.ln_1.weight), g(b.ln_1.bias),
             w[:C].ravel(), bb[:C], w[C:2*C].ravel(), bb[C:2*C], w[2*C:].ravel(), bb[2*C:],
             g(b.attn.c_proj.weight), g(b.attn.c_proj.bias),
             g(b.ln_2.weight), g(b.ln_2.bias),
             g(b.mlp.c_fc.weight), g(b.mlp.c_fc.bias),
             g(b.mlp.c_proj.weight), g(b.mlp.c_proj.bias),
             g(m.transformer.ln_f.weight), g(m.transformer.ln_f.bias)]
    return np.concatenate([p.astype(np.float32) for p in parts])


def field_sizes(C):
    F = 4 * C
    return [("wte", V * C), ("wpe", T * C), ("ln1g", C), ("ln1b", C),
            ("wq", C * C), ("bq", C), ("wk", C * C), ("bk", C), ("wv", C * C), ("bv", C),
            ("wo", C * C), ("bo", C), ("ln2g", C), ("ln2b", C),
            ("wfc", F * C), ("bfc", F), ("wmp", C * F), ("bmp", C), ("lnfg", C), ("lnfb", C)]


def f32lit(v):
    v = float(v)
    if v == 0.0:
        return "0.0"
    s = f"{abs(v):.12f}"
    return f"F32.neg({s})" if v < 0 else s


def vec_lit(vals, chunk=32):
    """Weights as a balanced concat tree of short literals (a single 4.6k-deep literal overflows the compiler stack)."""
    def small(vs):
        res = "VNil{}"
        for v in reversed(vs):
            res = f"VCon{{{f32lit(v)}, {res}}}"
        return res

    def tree(parts):
        if len(parts) == 1:
            return parts[0]
        mid = len(parts) // 2
        return f"vec_concat({tree(parts[:mid])}, {tree(parts[mid:])})"

    return tree([small(vals[i:i + chunk]) for i in range(0, len(vals), chunk)])


def quantize(flat):
    """Round-trip through the decimal literal so Bend and PyTorch start bit-identical."""
    return np.array([np.float32(float(f"{abs(float(v)):.12f}") * (-1 if v < 0 else 1)) for v in flat], dtype=np.float32)


# ------------------------------------------------------------------ Bend source
def gen_bend(C, depth, steps, print_every, flat):
    F = 4 * C
    sizes = field_sizes(C)
    nparam = sum(s for _, s in sizes)
    assert nparam == len(flat), (nparam, len(flat))
    B = 2 ** depth
    hdr = [f"def nV() -> Nat:\n  {V}n", f"def nC() -> Nat:\n  {C}n", f"def nF() -> Nat:\n  {F}n",
           f"def nT() -> Nat:\n  {T}n", f"def fscale() -> F32:\n  {f32lit(1/math.sqrt(C))}"]
    for n, s in sizes:
        hdr.append(f"def sz_{n}() -> Nat:\n  {s}n")
    header = "\n\n".join(hdr)

    shapes = {"wte": (V, C), "wpe": (T, C), "wq": (C, C), "wk": (C, C), "wv": (C, C), "wo": (C, C),
              "wfc": (F, C), "wmp": (C, F)}
    lines = ["def unflatten(+f: Vec) -> P:"]
    prev = "f"
    names = []
    for i, (n, s) in enumerate(sizes):
        nxt = f"r{i}"
        lines.append(f"  +{nxt} : Vec = vdrop(sz_{n}(), {prev})" if i < len(sizes) - 1 else "")
        names.append(n)
        prev = nxt
    # Written explicitly: piece_i = take(size_i, rest_{i-1}); rest_i = drop(size_i, rest_{i-1})
    lines = ["def unflatten(+f: Vec) -> P:"]
    rest = "f"
    exprs = []
    for i, (n, s) in enumerate(sizes):
        piece = f"vtake(sz_{n}(), {rest})"
        if n in shapes:
            r, c = shapes[n]
            piece = f"mat_of({r}n, {c}n, {piece})"
        lines.append(f"  +{n} : {'Mat' if n in shapes else 'Vec'} = {piece}")
        if i < len(sizes) - 1:
            lines.append(f"  +rest{i} : Vec = vdrop(sz_{n}(), {rest})")
            rest = f"rest{i}"
    lines.append("  P{" + ", ".join(n for n, _ in sizes) + "}")
    unflat = "\n".join(lines)

    # `rest` is used twice per step (take + drop) so it must be reusable (+f / +restN above)
    unflat = unflat.replace("def unflatten(+f: Vec)", "def unflatten(+f: Vec)")

    def toks(ids):
        r = "TNil{}"
        for t in reversed(ids):
            r = f"TCon{{{t}, {r}}}"
        return r

    def table(name, data):
        cases = "\n".join(f"    case {i}:\n      {toks(d)}" for i, d in enumerate(data[:-1]))
        return (f"def {name}_at(+i: U32) -> Toks:\n  match i:\n{cases}\n    case _:\n      {toks(data[-1])}\n\n"
                f"def {name}(+i: U32) -> Toks:\n  {name}_at(U32.mod(i, {len(data)}))")

    gen = f"""{table('phrase_x', X)}

{table('phrase_y', Y)}

{unflat}

def init_flat() -> Vec:
  {vec_lit(flat.tolist())}

def depth() -> Nat:
  {depth}n

def batch_size() -> U32:
  {B}

def inv_bt() -> F32:
  {f32lit(1.0 / (B * T))}

def n_steps() -> Nat:
  {steps}n

def n_params() -> Nat:
  {nparam}n

def print_every() -> U32:
  {print_every}

def adam_lr() -> F32:
  {f32lit(LR)}

def adam_b1() -> F32:
  {f32lit(B1)}

def adam_b2() -> F32:
  {f32lit(B2)}

def adam_eps() -> F32:
  {f32lit(EPS)}
"""
    matrix = open(os.path.join(ROOT, "src", "matrix.bend")).read().split("def main() -> IO(Unit):")[0].strip()
    lib = open(os.path.join(HERE, "train_lib.bend")).read()
    pre, post = lib.split("# @@GENERATED@@")
    return f"{matrix}\n\n{header}\n\n{pre}\n{gen}\n{post}"


# ------------------------------------------------------------------ run Bend
def env():
    e = dict(os.environ)
    if CLANG_BIN:
        e["PATH"] = CLANG_BIN + ":" + e["PATH"]
    # `!` builds link against libcuda and load NVRTC from the CUDA 12 toolkit
    e["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:" + e.get("LIBRARY_PATH", "")
    e["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + e.get("LD_LIBRARY_PATH", "")
    return e


def _unlimited_stack():
    import resource
    resource.setrlimit(resource.RLIMIT_STACK, (4 << 30, resource.RLIM_INFINITY))  # 'unlimited' breaks the thread stacks


def build_bend(C, depth, steps, print_every, flat, tag):
    os.makedirs(WORK, exist_ok=True)
    src = os.path.join(WORK, f"train_{tag}.bend")
    out = os.path.join(WORK, f"train_{tag}")
    open(src, "w").write(gen_bend(C, depth, steps, print_every, flat))
    t = time.time()
    p = subprocess.run([BEND, src, "-o", out], capture_output=True, text=True, env=env(),
                       preexec_fn=_unlimited_stack)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"bend build failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return out, time.time() - t


def run_bend(binary, gpu, threads=None, timeout=3600):
    cmd = [binary, "--gpu", "on" if gpu else "off"]
    if threads:
        cmd += ["--threads", str(threads)]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env(), timeout=timeout,
                       preexec_fn=_unlimited_stack)
    wall = time.time() - t
    losses = {int(m.group(1)): (float(m.group(2)), int(m.group(3)))
              for m in re.finditer(r"step (\d+) loss (\S+) t_ms (\d+)", p.stdout)}
    tot = re.search(r"total_ms (\d+)", p.stdout)
    return dict(losses=losses, total_ms=int(tot.group(1)) if tot else None, wall=wall,
                rc=p.returncode, out=p.stdout, err=p.stderr)


# ------------------------------------------------------------------ PyTorch reference
def run_torch(C, B, steps, device, print_every=1, seed=42, flat_override=None, warmup=0):
    m = make_model(C, seed)
    if flat_override is not None:  # make PyTorch start from the exact decimal-rounded values Bend sees
        load_flat(m, C, flat_override)
    m.to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0)
    xs = torch.tensor([X[i % 4] for i in range(B)], dtype=torch.long, device=device)
    ys = torch.tensor([Y[i % 4] for i in range(B)], dtype=torch.long, device=device)
    losses = {}

    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = m(xs, ys)
        loss.backward()
        opt.step()
        return loss

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):  # cuda context / cublas / allocator warm-up, not timed
        step()
    sync()
    t0 = time.perf_counter()
    for s in range(steps):
        loss = step()
        if print_every and s % print_every == 0:
            losses[s] = float(loss.item())  # also syncs
    sync()
    total = (time.perf_counter() - t0) * 1000
    return dict(losses=losses, total_ms=total, ms_per_step=total / steps)


def load_flat(m, C, flat):
    F = 4 * C
    f = torch.tensor(flat)
    o = [0]

    def take(n, shape):
        t = f[o[0]:o[0] + n].reshape(shape).clone()
        o[0] += n
        return t

    with torch.no_grad():
        b = m.transformer.h[0]
        m.transformer.wte.weight.copy_(take(V * C, (V, C)))
        m.transformer.wpe.weight[:T].copy_(take(T * C, (T, C)))
        b.ln_1.weight.copy_(take(C, (C,))); b.ln_1.bias.copy_(take(C, (C,)))
        wq = take(C * C, (C, C)); bq = take(C, (C,)); wk = take(C * C, (C, C)); bk = take(C, (C,))
        wv = take(C * C, (C, C)); bv = take(C, (C,))
        b.attn.c_attn.weight.copy_(torch.cat([wq, wk, wv], 0)); b.attn.c_attn.bias.copy_(torch.cat([bq, bk, bv]))
        b.attn.c_proj.weight.copy_(take(C * C, (C, C))); b.attn.c_proj.bias.copy_(take(C, (C,)))
        b.ln_2.weight.copy_(take(C, (C,))); b.ln_2.bias.copy_(take(C, (C,)))
        b.mlp.c_fc.weight.copy_(take(F * C, (F, C))); b.mlp.c_fc.bias.copy_(take(F, (F,)))
        b.mlp.c_proj.weight.copy_(take(C * F, (C, F))); b.mlp.c_proj.bias.copy_(take(C, (C,)))
        m.transformer.ln_f.weight.copy_(take(C, (C,))); m.transformer.ln_f.bias.copy_(take(C, (C,)))
    assert o[0] == len(flat)


# ------------------------------------------------------------------ commands
def cmd_verify(a):
    C, depth, steps = a.C, a.depth, a.steps
    B = 2 ** depth
    m = make_model(C)
    flat = quantize(flat_params(m, C))
    ref = run_torch(C, B, steps, "cpu", print_every=1, flat_override=flat)
    binary, bt = build_bend(C, depth, steps, 1, flat, f"verify_C{C}_d{depth}")
    print(f"bend build {bt:.1f}s; running ...", flush=True)
    r = run_bend(binary, gpu=a.gpu)
    if r["rc"] != 0:
        print(r["out"][-2000:], r["err"][-2000:])
    print(f"{'step':>5} {'torch loss':>12} {'bend loss':>12} {'abs diff':>10}")
    worst = 0
    for s in sorted(r["losses"]):
        bl = r["losses"][s][0]
        tl = ref["losses"][s]
        worst = max(worst, abs(bl - tl))
        if s < 5 or s % max(1, steps // 10) == 0 or s == steps - 1:
            print(f"{s:5d} {tl:12.6f} {bl:12.6f} {abs(bl - tl):10.2e}")
    print(f"max |diff| = {worst:.3e}  (bend total {r['total_ms']} ms)")


def bend_step_ms(r):
    """ms/step from the per-step timestamps, dropping step 0 (JIT / first-touch / CUDA init)."""
    ls = r["losses"]
    ks = sorted(ls)
    if len(ks) < 2:
        return None
    return (ls[ks[-1]][1] - ls[ks[0]][1]) / (ks[-1] - ks[0])


def cmd_bench(a):
    C = a.C
    m = make_model(C)
    flat = quantize(flat_params(m, C))
    rows = []
    for depth in a.depths:
        B = 2 ** depth
        row = dict(C=C, B=B)
        # --- PyTorch
        for dev in ("cpu", "cuda"):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            n = 20 if dev == "cpu" else 50
            r = run_torch(C, B, n, dev, print_every=0, warmup=5)
            row[f"torch_{dev}"] = r["ms_per_step"]
        # --- Bend
        steps = a.steps
        binary, bt = build_bend(C, depth, steps, 1, flat, f"bench_C{C}_d{depth}")
        for name, gpu, thr in (("bend_cpu1", False, 1), ("bend_cpu24", False, 24), ("bend_gpu", True, None)):
            if name == "bend_gpu" and depth < a.gpu_min_depth:
                continue
            try:
                r = run_bend(binary, gpu, thr, timeout=a.timeout)
                row[name] = bend_step_ms(r)
                if r["rc"] != 0:
                    row[name] = None
                    print("  !", name, r["err"][-150:].strip(), flush=True)
            except subprocess.TimeoutExpired:
                row[name] = None
                print("  ! timeout", name, flush=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "bench"])
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--gpu-min-depth", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"verify": cmd_verify, "bench": cmd_bench}[a.cmd](a)
