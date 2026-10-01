import Base

# Nat-counted tail recursion -> a for(;;) loop with a counter
def loop(+n: Nat, +i: U32, +acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n+p:
      loop(p, (i + 1 : U32), (acc + U32.to_f32(i) * 0.5 : F32))

def kern(+i: U32) -> F32:
  loop(8n, i, 0.0)
