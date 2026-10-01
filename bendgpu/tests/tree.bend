import Base

# a tree with data at the nodes, built and folded recursively; Bool and Nat matches inside
type Tr is Data:
  Leaf{v: F32}
  Node{l: Tr, r: Tr, w: F32}

def make(+d: Nat, +x: F32) -> Tr:
  match d:
    case 0n:
      Leaf{x}
    case 1n+p:
      Node{make(p, (x * 2.0 : F32)), make(p, (x * 2.0 + 1.0 : F32)), x}

def fold(+t: Tr) -> F32:
  match t:
    case Leaf{v}:
      v
    case Node{l, r, w}:
      (fold(l) * 0.5 + fold(r) * 0.25 + w : F32)

def kern(+i: U32) -> F32:
  fold(make(5n, U32.to_f32(U32.mod(i, 7))))
