import Base

type P is Data:
  P{a: F32, b: F32}

# a list of records built by non-tail recursion, then consumed with a nested pattern: both on the heap
def build(+n: Nat, +x: F32) -> +List<P>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      Con{P{x, (x * 2.0 : F32)}, build(p, (x + 1.0 : F32))}

def sum(+l: +List<P>, +acc: F32) -> F32:
  match l:
    case Nil{}:
      acc
    case Con{P{a, b}, t}:
      sum(t, (acc + a * b : F32))

def kern(+i: U32) -> F32:
  sum(build(6n, U32.to_f32(i)), 0.0)
