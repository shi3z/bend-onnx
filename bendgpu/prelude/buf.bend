import Base

# A read-only buffer of 2^d F32 values: a perfect binary tree, so it is `Data` (shareable by every
# thread; Base's `Array` is a `Type`, i.e. affine, and cannot be read by many lanes).
#
# This file is ordinary Bend: it type-checks and runs on the stock runtime, which makes it the
# reference semantics. The CUDA back end recognises `Buf.get` and compiles a `Buf` to a device
# pointer and a `get` to one load.
type Buf is Data:
  BLeaf{v: F32}
  BNode{l: Buf, r: Buf}

# element i of a buffer of 2^d elements (only the low d bits of i are used)
def Buf.get(+d: Nat, +b: Buf, +i: U32) -> F32:
  match d b:
    case 0n BLeaf{+v}:
      v
    case 1n+p BNode{l, r}:
      Buf.get(p, Bool.pick(Buf, U32.is_ne(U32.and(U32.shrn(i, p), 1), 0), r, l), i)
    case _ _:
      0.0

# the buffer whose element j is f(j), built on the stock runtime only to give tests a reference
def Buf.build(~f: U32 -> F32, +d: Nat, +off: U32) -> Buf:
  match d:
    case 0n:
      BLeaf{f(off)}
    case 1n+p:
      BNode{Buf.build(~f, p, off), Buf.build(~f, p, (off + U32.shln(1, p) : U32))}
