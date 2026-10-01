import Base

type P2 is Data:
  P2{x: F32, y: F32}

type Q is Data:
  Q{a: P2, b: F32}

def swap(p: P2) -> P2:
  match p:
    case P2{x, y}:
      P2{y, x}

def norm(+p: P2) -> F32:
  match p:
    case P2{x, y}:
      (x * x + y * y : F32)

def mix(q: Q) -> F32:
  match q:
    case Q{a, b}:
      (norm(swap(a)) * b : F32)

def kern(+i: U32) -> F32:
  mix(Q{P2{U32.to_f32(i), 2.0}, 0.5})
