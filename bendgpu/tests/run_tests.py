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


def kern_type(src):
    m = re.search(r"def kern\([^)]*\)\s*->\s*(\w+)", src)
    return m.group(1)


def kern_bufs(src):
    """names of the Buf parameters of kern (after the index) and the alias the file gave buf.bend"""
    sig = re.search(r"def kern\(([^)]*)\)", src).group(1)
    names = [re.sub(r"^\+", "", p.split(":")[0].strip()) for p in sig.split(",")][1:]
    alias = re.search(r"import\s+\S*buf\.bend\s+as\s+(\w+)", src)
    return names, (alias.group(1) if alias else None)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


bad = 0
for f in sorted(glob.glob(os.path.join(HERE, "*.bend"))):
    name = os.path.basename(f)[:-5]
    src = open(f).read()
    ty = kern_type(src)
    with tempfile.TemporaryDirectory() as d:
        cu, exe = os.path.join(d, name + ".cu"), os.path.join(d, name)
        r = run(node + [f, "kern"])
        if r.returncode:
            print(f"FAIL {name}: emit: {r.stderr.strip()[:200]}"); bad += 1; continue
        open(cu, "w").write(r.stdout)
        r = run([NVCC, "-O3", "-arch=sm_80", "--fmad=false", "-o", exe, cu])
        if not os.path.exists(exe):
            print(f"FAIL {name}: nvcc: {r.stderr.strip()[:300]}"); bad += 1; continue
        out = run([exe, "4096"]).stdout
        got = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"kern\((\d+)\) = (\S+)", out)}
        ref = {}
        idx = [0, 1, 2, 3, 1000, 1500, 2000, 2047, 3000, 4095]
        shown = "F32.show" if ty == "F32" else "U32.show"
        names, alias = kern_bufs(src)
        binds = "".join(f"    +{n} : {alias}.Buf = {alias}.Buf.build(~{n}_init, {n}_depth(), 0)\n" for n in names)
        args = "".join(f", {n}" for n in names)
        main = "\ndef main() -> IO(Unit):\n  do IO<Unit>:\n" + binds + "".join(f"    IO.print({shown}(kern({i}{args})))\n" for i in idx)
        rf = os.path.join(HERE, "_ref_" + name + ".bend")   # next to the test so ../prelude resolves
        open(rf, "w").write(src + main)
        rr = run([BEND, rf])
        os.remove(rf)
        vals = [float(x) for x in rr.stdout.split()] if rr.returncode == 0 else []
        ok = len(vals) == len(idx) and all(abs(v - got.get(i, float("nan"))) <= 1e-6 * max(1.0, abs(v)) for i, v in zip(idx, vals))
        print(("ok  " if ok else "FAIL"), name, "cuda", [got.get(i) for i in idx][:6], "bend", vals[:6],
              re.search(r"([\d.]+) ms per launch", out).group(1) + " ms/launch(4096)")
        bad += 0 if ok else 1
sys.exit(1 if bad else 0)
