import Base

def mag2(+zr: F32, +zi: F32) -> F32:
  (zr * zr + zi * zi : F32)

# escape-time iteration with a Bool pick and a Nat fuel
def iter(+n: Nat, +zr: F32, +zi: F32, +cr: F32, +ci: F32, +k: U32) -> U32:
  match n:
    case 0n:
      k
    case 1n+p:
      +esc : Bool = F32.is_gt(mag2(zr, zi), 4.0)
      iter(p, Bool.pick(F32, esc, zr, (zr * zr - zi * zi + cr : F32)), Bool.pick(F32, esc, zi, (2.0 * zr * zi + ci : F32)), cr, ci, Bool.pick(U32, esc, k, (k + 1 : U32)))

def kern(+i: U32) -> U32:
  +x : U32 = U32.mod(i, 64)
  +y : U32 = U32.div(i, 64)
  iter(200n, 0.0, 0.0, (U32.to_f32(x) * 0.046875 - 2.0 : F32), (U32.to_f32(y) * 0.046875 - 1.5 : F32), 0)
