#!/bin/bash
# emit and show where the checker complains
export BEND_SRC=/tmp/claude-1000/-home-shi3z-git-bend-onnx/6fc27097-9917-45ea-88b5-faaa132e6d53/scratchpad/bend-src
OUT=. ./run_emit.sh ${1:-4}
B=${1:-4}
python3 - <<PY
import re
e=open('err$B.txt').read()
m=re.search(r"\n  exp: (.*)\n  obs: (.*)\n",e)
if not m: print(e[-800:] if e.strip() else "emit ok"); raise SystemExit
print("exp:",m.group(1)[:200]); print("obs:",m.group(2)[:200])
m=re.search(r"\n    beg: (\d+)",e)
if m:
    b=int(m.group(1)); s=open('train$B.bend').read()
    d=s.rfind('\ndef ',0,b); print(s[d:d+500]); print('AT:',repr(s[b-80:b+80]))
PY
