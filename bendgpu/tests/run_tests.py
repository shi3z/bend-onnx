#!/usr/bin/env python3
"""For every tests/*.bend: emit CUDA, build, run, and compare kern(0..3) with Bend's own runtime.

  BEND_SRC=/path/to/bendlang/bend  python3 bendgpu/tests/run_tests.py
Needs node >= 22, nvcc, and the `bend` binary (BEND_BIN) for the reference values.
"""
import os, re, subprocess, sys, glob, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT = os.path.join(os.path.dirname(HERE), "emit_cuda.mjs")
BEND = os.environ.get("BEND_BIN", "/home/shi3z/snap/antigravity-cli/common/.bend/bin/bend")
NVCC = "/usr/local/cuda/bin/nvcc"
env = dict(os.environ)
node = ["node", "--experimental-transform-types", "--no-warnings", EMIT]


def entry_of(src):
    return "kern" if re.search(r"def kern\(", src) else "pipe"


def kern_type(src):
    m = re.search(r"def kern\([^)]*\)\s*->\s*(\w+)", src)
    return m.group(1) if m else "F32"   # a pipeline's output is a Buf of F32


def kern_bufs(src):
    """[(name, type text)] of the buffer parameters of the entry, and the alias the file gave buf.bend / arr.bend"""
    entry = entry_of(src)
    sig = re.search(rf"def {entry}\(([^)]*(?:<[^)]*>[^)]*)*)\)\s*->", src).group(1)
    parts, depth, cur = [], 0, ""
    for ch in sig:                      # split on top-level commas (types contain commas inside <>)
        depth += ch == "<"; depth -= ch == ">"
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    ps = [(re.sub(r"^\+", "", p.split(":", 1)[0].strip()), p.split(":", 1)[1].strip()) for p in parts]
    if entry == "kern":
        ps = ps[1:]
    alias = re.search(r"import\s+\S*(?:buf|arr)\.bend\s+as\s+(\w+)", src)
    return ps, (alias.group(1) if alias else None)


def out_type(src):
    return re.search(r"def pipe\([^\n]*\)\s*->\s*(.*):\s*$", src, re.M).group(1).strip()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


bad = 0
for f in sorted(glob.glob(os.path.join(HERE, "*.bend"))):
    name = os.path.basename(f)[:-5]
    src = open(f).read()
    ty = kern_type(src)
    with tempfile.TemporaryDirectory() as d:
        cu, exe = os.path.join(d, name + ".cu"), os.path.join(d, name)
        r = run(node + [f, entry_of(src)])
        if r.returncode:
            print(f"FAIL {name}: emit: {r.stderr.strip()[:200]}"); bad += 1; continue
        open(cu, "w").write(r.stdout)
        r = run([NVCC, "-O3", "-arch=sm_80", "--fmad=false", "-o", exe, cu])
        if not os.path.exists(exe):
            print(f"FAIL {name}: nvcc: {r.stderr.strip()[:300]}"); bad += 1; continue
        out = run([exe, "4096"]).stdout if entry_of(src) == "kern" else run([exe]).stdout
        got = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"kern\((\d+)\) = (\S+)", out)}
        ref = {}
        idx = [0, 1, 2, 3, 1000, 1500, 2000, 2047, 3000, 4095]
        shown = "F32.show" if ty == "F32" else "U32.show"
        ps, alias = kern_bufs(src)
        names = [n for n, _ in ps]

        def bind(n, ty):
            if "Arr<" in ty:
                elem = re.search(r"Arr<[^,]*,\s*(.*)>", ty).group(1)
                return f"    +{n} : {ty} = {alias}.Arr.build(~{elem}, ~{n}_init, {n}_depth(), 0)\n"
            return f"    +{n} : {alias}.Buf = {alias}.Buf.build(~{n}_init, {n}_depth(), 0)\n"
        binds = "".join(bind(n, ty) for n, ty in ps)
        args = "".join(f", {n}" for n in names)
        if entry_of(src) == "kern":
            main = "\ndef main() -> IO(Unit):\n  do IO<Unit>:\n" + binds + "".join(f"    IO.print({shown}(kern({i}{args})))\n" for i in idx)
        else:
            oty = out_type(src)
            if "Arr<" in oty:
                get = lambda i: f"{alias}.Arr.get(F32, pipe_depth(), y, {i}, 0.0)"
                yty = oty
            else:
                get = lambda i: f"{alias}.Buf.get(pipe_depth(), y, {i})"
                yty = f"{alias}.Buf"
            main = ("\ndef main() -> IO(Unit):\n  do IO<Unit>:\n" + binds + f"    +y : {yty} = pipe({args[2:]})\n"
                    + "".join(f"    IO.print(F32.show({get(i)}))\n" for i in idx))
        rf = os.path.join(HERE, "_ref_" + name + ".bend")   # next to the test so ../prelude resolves
        open(rf, "w").write(src + main)
        rr = run([BEND, rf])
        os.remove(rf)
        vals = [float(x) for x in rr.stdout.split()] if rr.returncode == 0 else []
        ok = len(vals) == len(idx) and all(abs(v - got.get(i, float("nan"))) <= 1e-6 * max(1.0, abs(v)) for i, v in zip(idx, vals))
        print(("ok  " if ok else "FAIL"), name, "cuda", [got.get(i) for i in idx][:6], "bend", vals[:6],
              re.search(r"([\d.]+) ms per (?:launch|pipeline run)", out).group(1) + " ms")
        bad += 0 if ok else 1
sys.exit(1 if bad else 0)
