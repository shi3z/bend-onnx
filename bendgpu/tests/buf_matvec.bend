import Base
import ../prelude/buf.bend as B

# W: 64 rows x 16 columns, x: 16 values; row i of the product, with a gather on the row index
def w_depth() -> Nat:
  10n

def w_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(U32.mul(j, 37), 101)) * 0.01 - 0.5 : F32)

def x_depth() -> Nat:
  4n

def x_init(j: U32) -> F32:
  (U32.to_f32(j) * 0.125 - 1.0 : F32)

def row(+n: Nat, +k: U32, +base: U32, +w: B.Buf, +x: B.Buf, +acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      row(p, (k + 1 : U32), base, w, x, (acc + B.Buf.get(10n, w, (base * 16 + k : U32)) * B.Buf.get(4n, x, k) : F32))

def kern(+i: U32, +w: B.Buf, +x: B.Buf) -> F32:
  row(16n, 0, U32.mod(U32.mul(i, 7), 64), w, x, 0.0)
