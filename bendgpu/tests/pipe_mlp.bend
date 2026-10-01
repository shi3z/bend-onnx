import Base
import ../prelude/buf.bend as B

# a two-stage pipeline: h = tanh(W1 x) (32 values), y[i] = W2[i % 64] . h (4096 values); h stays on the device
def w1_depth() -> Nat:
  9n

def w1_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(U32.mul(j, 37), 101)) * 0.02 - 1.0 : F32)

def x_depth() -> Nat:
  4n

def x_init(j: U32) -> F32:
  (U32.to_f32(j) * 0.125 - 1.0 : F32)

def w2_depth() -> Nat:
  11n

def w2_init(j: U32) -> F32:
  (U32.to_f32(U32.mod(U32.mul(j, 53), 97)) * 0.01 - 0.4 : F32)

def dot(+n: Nat, +k: U32, +wbase: U32, +w: B.Buf, +wd: Nat, +v: B.Buf, +vd: Nat, +acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      dot(p, (k + 1 : U32), wbase, w, wd, v, vd, (acc + B.Buf.get(wd, w, (wbase + k : U32)) * B.Buf.get(vd, v, k) : F32))

def s1(+i: U32, +w1: B.Buf, +x: B.Buf) -> F32:
  F32.tanh(dot(16n, 0, (i * 16 : U32), w1, 9n, x, 4n, 0.0))

def s2(+i: U32, +w2: B.Buf, +h: B.Buf) -> F32:
  dot(32n, 0, (U32.mod(i, 64) * 32 : U32), w2, 11n, h, 5n, 0.0)

def pipe_depth() -> Nat:
  12n

def pipe(+w1: B.Buf, +x: B.Buf, +w2: B.Buf) -> B.Buf:
  +h : B.Buf = B.Buf.map2(~s1, 5n, w1, x, 0)
  B.Buf.map2(~s2, 12n, w2, h, 0)
