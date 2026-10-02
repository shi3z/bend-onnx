#!/usr/bin/env python3
"""Generate, emit, build and run the nanoGPT training step on the bendgpu back end; verify the loss against PyTorch.

  python3 bendgpu/nanogpt/run.py verify [--B 4] [--steps 30]
  python3 bendgpu/nanogpt/run.py bench  [--Bs 4096,65536]
"""
import os, sys, subprocess, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "nanogpt"))
import bench_train as b

WORK = os.environ.get("BEND_WORK", "/tmp/bend_train")
os.makedirs(WORK, exist_ok=True)
NVCC = "/usr/local/cuda/bin/nvcc"
env = dict(os.environ)
env.setdefault("BEND_SRC", "/tmp/claude-1000/-home-shi3z-git-bend-onnx/6fc27097-9917-45ea-88b5-faaa132e6d53/scratchpad/bend-src")


def build(B):
    src, cu, exe = (os.path.join(WORK, f"bg_train{B}{e}") for e in (".bend", ".cu", ""))
    open(src, "w").write(subprocess.run([sys.executable, os.path.join(HERE, "gen.py"), str(B)], capture_output=True, text=True, check=True).stdout.replace("import ../prelude/arr.bend", "import " + os.path.join(os.path.dirname(HERE), "prelude/arr.bend")))
    r = subprocess.run(["node", "--experimental-transform-types", "--no-warnings", os.path.join(os.path.dirname(HERE), "emit_cuda.mjs"), src, "train"], capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("emit failed:\n" + r.stderr[-1500:])
    open(cu, "w").write(r.stdout)
    r = subprocess.run([NVCC, "-O3", "-arch=sm_80", "--fmad=false", "-o", exe, cu], capture_output=True, text=True)
    if r.returncode:
        sys.exit("nvcc failed:\n" + r.stderr[-3000:])
    return exe


def params():
    m = b.make_model(16)
    flat = b.quantize(b.flat_params(m, 16)).astype(np.float32)
    pf = os.path.join(WORK, "params.bin")
    s = np.zeros(16384, np.float32); s[:flat.size] = flat
    s.tofile(pf)
    return flat, pf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--B", type=int, default=4); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--Bs", default="4096,65536")
    a = ap.parse_args()
    flat, pf = params()
    if a.cmd == "verify":
        exe = build(a.B)
        ref = b.run_torch(16, a.B, a.steps, "cpu", print_every=1, flat_override=flat)
        out = subprocess.run([exe, str(a.steps), "12289", pf, "verify"], capture_output=True, text=True).stdout
        ls = {int(l.split()[1]): float(l.split()[3]) for l in out.strip().splitlines()}
        d = max(abs(ls[s] - ref["losses"][s]) for s in ls)
        print("steps", len(ls), "first losses", [round(ls[k], 5) for k in sorted(ls)[:3]], "torch", [round(ref["losses"][k], 5) for k in range(3)])
        print("max |loss diff| vs PyTorch:", d)
    else:
        for B in map(int, a.Bs.split(",")):
            exe = build(B)
            best = min(float(subprocess.run([exe, "20", "12289", pf], capture_output=True, text=True).stdout.split()[1]) for _ in range(3))
            print(f"B={B:7d}  bendgpu {best:9.4f} ms/step", flush=True)
