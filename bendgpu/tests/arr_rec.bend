import Base
import ../prelude/arr.bend as A

# arrays of records on the device: s1 builds P records, s2 gathers from them and builds Q records, s3 reduces to F32
type P is Data:
  P{x: F32, y: F32}

type Q is Data:
  Q{u: F32, v: F32}

def a_depth() -> Nat:
  4n

def a_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(j, 5)) * 0.5 - 1.0 : F32)

def s1(+i: U32, +a: A.Arr<&2, F32>) -> P:
  P{A.Arr.get(F32, 4n, a, i, 0.0), (U32.to_f32(U32.mod(i, 13)) * 0.1 : F32)}

def mixq(+p: P, +r: P) -> Q:
  match p r:
    case P{x, y} P{x2, y2}:
      Q{(x * x2 + y : F32), (y * y2 - x : F32)}

def s2(+i: U32, +ps: A.Arr<&2, P>) -> Q:
  mixq(A.Arr.get(P, 10n, ps, i, P{0.0, 0.0}), A.Arr.get(P, 10n, ps, (i * 7 : U32), P{0.0, 0.0}))

def qsum(+q: Q) -> F32:
  match q:
    case Q{u, v}:
      (u + 2.0 * v : F32)

def s3(+i: U32, +qs: A.Arr<&2, Q>) -> F32:
  qsum(A.Arr.get(Q, 12n, qs, i, Q{0.0, 0.0}))

def pipe_depth() -> Nat:
  12n

def pipe(+a: A.Arr<&2, F32>) -> A.Arr<&2, F32>:
  +ps : A.Arr<&2, P> = A.Arr.map1(~F32, ~P, ~s1, 10n, a, 0)
  +qs : A.Arr<&2, Q> = A.Arr.map1(~P, ~Q, ~s2, 12n, ps, 0)
  A.Arr.map1(~Q, ~F32, ~s3, 12n, qs, 0)
