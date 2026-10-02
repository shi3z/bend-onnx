#!/bin/bash
# usage: run_emit.sh B  -> train_B.bend / train_B.cu in $OUT
B=${1:-4}; OUT=${OUT:-.}
cd "$(dirname "$0")"
python3 gen.py $B > $OUT/train$B.bend && node --experimental-transform-types --no-warnings ../emit_cuda.mjs $OUT/train$B.bend train > $OUT/train$B.cu 2> $OUT/err$B.txt
