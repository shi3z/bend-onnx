import Base
import ../prelude/arr.bend as A

# nested records, fields used more than once, record arrays between stages
type V2 is Data:
  V2{a: F32, b: F32}

type W is Data:
  W{p: V2, q: V2, k: U32}

def a_depth() -> Nat:
  12n

def a_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(j, 7)) * 0.25 - 0.5 : F32)

def s1(+i: U32, +a: A.Arr<&2, F32>) -> W:
  +x : F32 = A.Arr.get(F32, 12n, a, i, 0.0)
  W{V2{x, (x * x : F32)}, V2{(x + 1.0 : F32), (x - 1.0 : F32)}, i}

def dot2(+u: V2, +v: V2) -> F32:
  match u v:
    case V2{a, b} V2{c, d}:
      (a * c + b * d + a * a : F32)

def fin(+w: W) -> F32:
  match w:
    case W{p, q, k}:
      (dot2(p, q) + dot2(q, p) + U32.to_f32(k) : F32)

def s2(+i: U32, +ws: A.Arr<&2, W>) -> F32:
  fin(A.Arr.get(W, 12n, ws, i, W{V2{0.0, 0.0}, V2{0.0, 0.0}, 0}))

def pipe_depth() -> Nat:
  12n

def pipe(+a: A.Arr<&2, F32>) -> A.Arr<&2, F32>:
  +ws : A.Arr<&2, W> = A.Arr.map1(~F32, ~W, ~s1, 12n, a, 0)
  A.Arr.map1(~W, ~F32, ~s2, 12n, ws, 0)
