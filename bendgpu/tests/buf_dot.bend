import Base
import ../prelude/buf.bend as B

def x_depth() -> Nat:
  6n

def x_init(j: U32) -> F32:
  (U32.to_f32(j) * 0.25 + 1.0 : F32)

def y_depth() -> Nat:
  4n

def y_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(j, 5)) - 2.0 : F32)

def dot(+n: Nat, +k: U32, +i: U32, +x: B.Buf, +y: B.Buf, +acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      dot(p, (k + 1 : U32), i, x, y, (acc + B.Buf.get(6n, x, (i + k : U32)) * B.Buf.get(4n, y, k) : F32))

def kern(+i: U32, +x: B.Buf, +y: B.Buf) -> F32:
  dot(16n, 0, i, x, y, 0.0)
