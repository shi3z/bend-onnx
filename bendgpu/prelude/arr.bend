import Base

# A read-only array of 2^d values of any Data type A (scalars, records of scalars, ...): a perfect binary tree, so it
# is `Data` and every thread can read it. Ordinary Bend: it runs on the stock runtime, which is the reference
# semantics. The CUDA back end compiles an `Arr<&2, A>` to a device pointer to A, `Arr.get` to one load and
# `Arr.mapN` to one kernel launch whose result array stays on the device (an array of records is an array of structs).
type Arr<a, -A: Kind(a)> is Kind(a):
  ArrLeaf{v: A}
  ArrNode{l: Arr<a, A>, r: Arr<a, A>}

# element i of an array of 2^d elements (only the low d bits of i are used); `z` is returned for a malformed tree
def Arr.get(-A: Data, +d: Nat, +b: Arr<&2, A>, +i: U32, +z: A) -> A:
  match d b:
    case 0n ArrLeaf{+v}:
      v
    case 1n+p ArrNode{l, r}:
      Arr.get(A, p, Bool.pick(Arr<&2, A>, U32.is_ne(U32.and(U32.shrn(i, p), 1), 0), r, l), i, z)
    case _ _:
      z

# the array whose element j is f(j) (reference only)
def Arr.build(~A: Data, ~f: U32 -> A, +d: Nat, +off: U32) -> Arr<&2, A>:
  match d:
    case 0n:
      ArrLeaf{f(off)}
    case 1n+p:
      ArrNode{Arr.build(~A, ~f, p, off), Arr.build(~A, ~f, p, (off + U32.shln(1, p) : U32))}

def Arr.map1(~A0: Data, ~C: Data, ~f: @+i:U32 -> @+b0:Arr<&2, A0> -> C, +d: Nat, +b0: Arr<&2, A0>, +off: U32) -> Arr<&2, C>:
  match d:
    case 0n:
      ArrLeaf{f(off, b0)}
    case 1n+p:
      l r = Arr.map1(~A0, ~C, ~f, p, b0, off) Arr.map1(~A0, ~C, ~f, p, b0, (off + U32.shln(1, p) : U32))
      ArrNode{l, r}

def Arr.map2(~A0: Data, ~A1: Data, ~C: Data, ~f: @+i:U32 -> @+b0:Arr<&2, A0> -> @+b1:Arr<&2, A1> -> C, +d: Nat, +b0: Arr<&2, A0>, +b1: Arr<&2, A1>, +off: U32) -> Arr<&2, C>:
  match d:
    case 0n:
      ArrLeaf{f(off, b0, b1)}
    case 1n+p:
      l r = Arr.map2(~A0, ~A1, ~C, ~f, p, b0, b1, off) Arr.map2(~A0, ~A1, ~C, ~f, p, b0, b1, (off + U32.shln(1, p) : U32))
      ArrNode{l, r}

def Arr.map3(~A0: Data, ~A1: Data, ~A2: Data, ~C: Data, ~f: @+i:U32 -> @+b0:Arr<&2, A0> -> @+b1:Arr<&2, A1> -> @+b2:Arr<&2, A2> -> C, +d: Nat, +b0: Arr<&2, A0>, +b1: Arr<&2, A1>, +b2: Arr<&2, A2>, +off: U32) -> Arr<&2, C>:
  match d:
    case 0n:
      ArrLeaf{f(off, b0, b1, b2)}
    case 1n+p:
      l r = Arr.map3(~A0, ~A1, ~A2, ~C, ~f, p, b0, b1, b2, off) Arr.map3(~A0, ~A1, ~A2, ~C, ~f, p, b0, b1, b2, (off + U32.shln(1, p) : U32))
      ArrNode{l, r}

def Arr.map4(~A0: Data, ~A1: Data, ~A2: Data, ~A3: Data, ~C: Data, ~f: @+i:U32 -> @+b0:Arr<&2, A0> -> @+b1:Arr<&2, A1> -> @+b2:Arr<&2, A2> -> @+b3:Arr<&2, A3> -> C, +d: Nat, +b0: Arr<&2, A0>, +b1: Arr<&2, A1>, +b2: Arr<&2, A2>, +b3: Arr<&2, A3>, +off: U32) -> Arr<&2, C>:
  match d:
    case 0n:
      ArrLeaf{f(off, b0, b1, b2, b3)}
    case 1n+p:
      l r = Arr.map4(~A0, ~A1, ~A2, ~A3, ~C, ~f, p, b0, b1, b2, b3, off) Arr.map4(~A0, ~A1, ~A2, ~A3, ~C, ~f, p, b0, b1, b2, b3, (off + U32.shln(1, p) : U32))
      ArrNode{l, r}
