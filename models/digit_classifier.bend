#!/usr/bin/env bend
# Auto-generated Bend neural network compiled from ONNX
# Model: deep_digit_classifier
import Base

# === Bend Matrix & Tensor Core Primitives ===
type Vec is Data:
  VNil{}
  VCon{head: F32, tail: Vec}

def vec_dot_acc(x: Vec, y: Vec, acc: F32) -> F32:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      vec_dot_acc(xt, yt, (acc + xh * yh : F32))
    case _ _:
      acc

def vec_dot(x: Vec, y: Vec) -> F32:
  vec_dot_acc(x, y, 0.0)

def vec_add(x: Vec, y: Vec) -> Vec:
  match x y:
    case VCon{+xh, xt} VCon{+yh, yt}:
      VCon{(xh + yh : F32), vec_add(xt, yt)}
    case _ _:
      VNil{}

def vec_concat(x: Vec, y: Vec) -> Vec:
  match x:
    case VNil{}:
      y
    case VCon{h, t}:
      VCon{h, vec_concat(t, y)}

def vec_relu(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{F32.max(0.0, h), vec_relu(t)}

def f32_sigmoid(x: F32) -> F32:
  neg_x = F32.neg(x)
  e = F32.exp(neg_x)
  (1.0 / (1.0 + e : F32) : F32)

def vec_sigmoid(v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{f32_sigmoid(h), vec_sigmoid(t)}

def vec_max_acc(v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vec_max_acc(t, F32.max(acc, h))

def vec_max(v: Vec) -> F32:
  match v:
    case VNil{}:
      0.0
    case VCon{+h, t}:
      vec_max_acc(t, h)

def vec_exp_shifted(+shift: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{F32.exp((h - shift : F32)), vec_exp_shifted(shift, t)}

def vec_sum_acc(v: Vec, acc: F32) -> F32:
  match v:
    case VNil{}:
      acc
    case VCon{+h, t}:
      vec_sum_acc(t, (acc + h : F32))

def vec_sum(v: Vec) -> F32:
  vec_sum_acc(v, 0.0)

def vec_scale_div(+denom: F32, v: Vec) -> Vec:
  match v:
    case VNil{}:
      VNil{}
    case VCon{+h, t}:
      VCon{(h / denom : F32), vec_scale_div(denom, t)}

def vec_softmax(+v: Vec) -> Vec:
  m : F32 = vec_max(v)
  +exps : Vec = vec_exp_shifted(m, v)
  s : F32 = vec_sum(exps)
  vec_scale_div(s, exps)

def vec_argmax_go(v: Vec, +best_idx: U32, +best_val: F32, +curr_idx: U32) -> U32:
  match v:
    case VNil{}:
      best_idx
    case VCon{+h, t}:
      +is_better = F32.is_gt(h, best_val)
      new_idx = Bool.pick(U32, is_better, curr_idx, best_idx)
      new_val = Bool.pick(F32, is_better, h, best_val)
      vec_argmax_go(t, new_idx, new_val, (curr_idx + 1 : U32))

def vec_argmax(v: Vec) -> U32:
  match v:
    case VNil{}:
      0
    case VCon{+h, t}:
      vec_argmax_go(t, 0, h, 1)

type MatTree is Data:
  MTLeaf{vec: Vec}
  MTNode{left: MatTree, right: MatTree}

def mat_tree_mul(+x: Vec, m: MatTree) -> Vec:
  match m:
    case MTLeaf{row}:
      VCon{vec_dot(row, x), VNil{}}
    case MTNode{left, right}:
      l r = mat_tree_mul!(x, left) mat_tree_mul!(x, right)
      vec_concat(l, r)

def linear_layer(+x: Vec, w: MatTree, b: Vec) -> Vec:
  raw : Vec = mat_tree_mul(x, w)
  vec_add(raw, b)

def vec_show_inner(v: Vec) -> String:
  match v:
    case VNil{}:
      ""
    case VCon{+h, t}:
      " " ++ F32.show(h) ++ vec_show_inner(t)

def vec_show(v: Vec) -> String:
  "[" ++ vec_show_inner(v) ++ " ]"

# === Model Parameters (Trained Weights & Biases) ===

def param_W1() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.03814735, VCon{0.4205673, VCon{0.0944445, VCon{F32.neg(0.2575347), VCon{F32.neg(0.07984033), VCon{F32.neg(0.1946702), VCon{0.4698803, VCon{F32.neg(0.6274111), VCon{0.4368984, VCon{0.283588, VCon{F32.neg(0.1200604), VCon{0.09456819, VCon{F32.neg(0.3330182), VCon{F32.neg(0.1754465), VCon{F32.neg(0.05652287), VCon{0.2439071, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.04839142), VCon{0.4802615, VCon{0.2953028, VCon{F32.neg(0.2506342), VCon{F32.neg(0.0559948), VCon{F32.neg(0.2574201), VCon{F32.neg(0.2279335), VCon{F32.neg(0.4536163), VCon{0.5243324, VCon{0.6390165, VCon{0.02134627, VCon{F32.neg(0.5524927), VCon{F32.neg(0.1057772), VCon{F32.neg(0.1226711), VCon{0.1677931, VCon{0.1422039, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.51079), VCon{F32.neg(0.392486), VCon{0.1274495, VCon{0.002000605, VCon{F32.neg(0.141205), VCon{0.1441566, VCon{0.3198559, VCon{0.3468955, VCon{F32.neg(0.0007229255), VCon{0.02077345, VCon{0.2541072, VCon{0.1985053, VCon{F32.neg(0.2124701), VCon{F32.neg(0.1495816), VCon{0.1814627, VCon{0.09016294, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.0753686), VCon{0.166833, VCon{0.1663818, VCon{0.02988901, VCon{F32.neg(0.2363017), VCon{0.1893028, VCon{0.3853471, VCon{F32.neg(0.060667), VCon{F32.neg(0.4694439), VCon{0.5499489, VCon{F32.neg(0.03961383), VCon{0.2816382, VCon{0.5494949, VCon{0.2649395, VCon{F32.neg(0.1461115), VCon{0.2733871, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3446942), VCon{F32.neg(0.1348101), VCon{F32.neg(0.1235818), VCon{0.1299452, VCon{F32.neg(0.02824472), VCon{0.747338, VCon{F32.neg(0.3043955), VCon{0.2654197, VCon{0.364183, VCon{0.4521425, VCon{F32.neg(0.131506), VCon{0.3095908, VCon{0.04556517, VCon{0.3249849, VCon{0.09126505, VCon{0.1440449, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.0176839), VCon{F32.neg(0.4391128), VCon{0.3025857, VCon{0.0741404, VCon{F32.neg(0.4277011), VCon{0.4328221, VCon{0.3977706, VCon{0.3240762, VCon{0.6207566, VCon{F32.neg(0.09061568), VCon{F32.neg(0.1433276), VCon{0.5448857, VCon{F32.neg(0.2694497), VCon{0.01620812, VCon{0.1838257, VCon{0.0765183, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.03267881, VCon{F32.neg(0.01173815), VCon{F32.neg(0.3237667), VCon{0.05755151, VCon{0.6406595, VCon{F32.neg(0.06899283), VCon{0.02190128, VCon{0.2204789, VCon{F32.neg(0.2783031), VCon{F32.neg(0.2140828), VCon{F32.neg(0.005313089), VCon{0.5105531, VCon{F32.neg(0.3564709), VCon{F32.neg(0.5052986), VCon{0.1314502, VCon{0.05644403, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.4691533), VCon{F32.neg(0.3635982), VCon{F32.neg(0.3304358), VCon{F32.neg(0.128312), VCon{0.1440782, VCon{F32.neg(0.3255755), VCon{0.0637494, VCon{F32.neg(0.07411364), VCon{0.1124971, VCon{F32.neg(0.2136373), VCon{0.01352937, VCon{0.2577259, VCon{0.4141772, VCon{F32.neg(0.1537504), VCon{0.1996954, VCon{0.3235973, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.3661053, VCon{0.1074282, VCon{0.6173775, VCon{F32.neg(0.1517687), VCon{0.1804823, VCon{0.006618481, VCon{F32.neg(0.1132294), VCon{0.3597522, VCon{F32.neg(0.09215974), VCon{0.1572434, VCon{F32.neg(0.490704), VCon{0.08428366, VCon{F32.neg(0.3842332), VCon{0.9211751, VCon{0.004220939, VCon{0.1267301, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3020446), VCon{F32.neg(0.2120488), VCon{F32.neg(0.6208746), VCon{0.3131931, VCon{0.0006612752, VCon{F32.neg(0.2619931), VCon{0.4026108, VCon{F32.neg(0.2197261), VCon{F32.neg(0.05962937), VCon{0.02394333, VCon{0.06790839, VCon{F32.neg(0.207687), VCon{F32.neg(0.2216164), VCon{0.2854406, VCon{F32.neg(0.3173174), VCon{F32.neg(0.2780567), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1148237), VCon{F32.neg(0.06003528), VCon{F32.neg(0.05049867), VCon{0.004199277, VCon{0.2051278, VCon{0.1440392, VCon{F32.neg(0.09261335), VCon{F32.neg(0.1554912), VCon{0.06335697, VCon{F32.neg(0.3613779), VCon{0.007035173, VCon{0.222212, VCon{0.2504937, VCon{F32.neg(0.2177578), VCon{F32.neg(0.3894905), VCon{F32.neg(0.3408879), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1487018), VCon{0.178433, VCon{F32.neg(0.1274986), VCon{F32.neg(0.1741443), VCon{F32.neg(0.1735148), VCon{F32.neg(0.2373931), VCon{0.3205659, VCon{F32.neg(0.1501649), VCon{0.2687222, VCon{F32.neg(0.1454153), VCon{0.02597965, VCon{F32.neg(0.1512003), VCon{0.4914928, VCon{0.4893102, VCon{0.2317969, VCon{0.08935878, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.2291219, VCon{F32.neg(0.5707427), VCon{0.2376645, VCon{0.1721473, VCon{0.08289421, VCon{F32.neg(0.000839984), VCon{0.2697624, VCon{F32.neg(0.07575819), VCon{F32.neg(0.9515281), VCon{0.3401615, VCon{0.0406875, VCon{F32.neg(0.1661634), VCon{F32.neg(0.0460005), VCon{F32.neg(0.2409626), VCon{0.04406504, VCon{F32.neg(0.2006454), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1211821, VCon{0.2408908, VCon{F32.neg(0.2861841), VCon{0.07613657, VCon{0.05220105, VCon{0.107027, VCon{F32.neg(0.4200613), VCon{F32.neg(0.2657832), VCon{F32.neg(0.4403698), VCon{0.3702454, VCon{F32.neg(0.3422871), VCon{F32.neg(0.8527908), VCon{0.1703254, VCon{F32.neg(0.2611323), VCon{0.008505465, VCon{F32.neg(0.1807936), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2277027, VCon{0.1979908, VCon{0.5837597, VCon{F32.neg(0.2130229), VCon{F32.neg(0.3736418), VCon{F32.neg(0.5021239), VCon{F32.neg(0.1948426), VCon{F32.neg(0.1106981), VCon{F32.neg(0.1926197), VCon{F32.neg(0.008690217), VCon{F32.neg(0.4140374), VCon{F32.neg(0.3333581), VCon{F32.neg(0.3589548), VCon{F32.neg(0.312848), VCon{F32.neg(0.3026873), VCon{F32.neg(0.4239872), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4901883, VCon{F32.neg(0.001400406), VCon{F32.neg(0.1575192), VCon{0.07075773, VCon{0.5526679, VCon{0.3645493, VCon{0.412748, VCon{0.08229943, VCon{F32.neg(0.1019416), VCon{F32.neg(0.04425266), VCon{0.1475522, VCon{0.07729991, VCon{F32.neg(0.2549815), VCon{F32.neg(0.1368202), VCon{F32.neg(0.2937921), VCon{0.0001994284, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3016714), VCon{F32.neg(0.1607365), VCon{0.1550322, VCon{F32.neg(0.7109113), VCon{F32.neg(0.1950461), VCon{0.0764033, VCon{F32.neg(0.1610565), VCon{0.2670815, VCon{F32.neg(0.2560298), VCon{F32.neg(0.05759186), VCon{0.4097069, VCon{F32.neg(0.2041979), VCon{0.1170527, VCon{0.06319694, VCon{0.1181343, VCon{F32.neg(0.018163), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2073798, VCon{0.05134255, VCon{F32.neg(0.1060186), VCon{0.06544239, VCon{0.0623095, VCon{F32.neg(0.2380417), VCon{0.0380916, VCon{F32.neg(0.06494138), VCon{F32.neg(0.006085584), VCon{0.2480873, VCon{F32.neg(0.1901264), VCon{F32.neg(0.008798935), VCon{F32.neg(0.1799265), VCon{0.3609649, VCon{F32.neg(0.3850861), VCon{0.2488706, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.3756288, VCon{F32.neg(0.06556198), VCon{F32.neg(0.1948704), VCon{0.3486727, VCon{F32.neg(0.1140469), VCon{F32.neg(0.1209064), VCon{0.1643435, VCon{F32.neg(0.1391262), VCon{F32.neg(0.6937903), VCon{F32.neg(0.0785552), VCon{F32.neg(0.7107955), VCon{0.01619397, VCon{F32.neg(0.3953033), VCon{0.2982882, VCon{F32.neg(0.1764233), VCon{0.3356671, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.3444791), VCon{F32.neg(0.1750241), VCon{0.2282849, VCon{0.2743274, VCon{F32.neg(0.01106976), VCon{F32.neg(0.4142611), VCon{0.3476243, VCon{F32.neg(0.2113502), VCon{0.7839692, VCon{F32.neg(0.3559413), VCon{F32.neg(0.569269), VCon{0.14359, VCon{F32.neg(0.009498924), VCon{F32.neg(0.001516237), VCon{0.02202715, VCon{0.1182554, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.2857545), VCon{F32.neg(0.1677931), VCon{0.06024637, VCon{0.2428056, VCon{0.1248128, VCon{F32.neg(0.3289543), VCon{F32.neg(0.1143234), VCon{0.03948029, VCon{F32.neg(0.3030919), VCon{F32.neg(0.05038713), VCon{0.1906298, VCon{0.1579769, VCon{F32.neg(0.0009544723), VCon{0.9337699, VCon{0.1895943, VCon{F32.neg(1.052281), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.01761099, VCon{0.4663795, VCon{0.3852267, VCon{F32.neg(0.4029243), VCon{0.2655208, VCon{0.2756969, VCon{0.2138906, VCon{F32.neg(0.6136566), VCon{0.1561029, VCon{F32.neg(0.09465152), VCon{F32.neg(0.3111027), VCon{0.2465274, VCon{0.6469955, VCon{0.4830784, VCon{F32.neg(0.1387634), VCon{0.101794, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2749307), VCon{F32.neg(0.4026341), VCon{0.1594216, VCon{F32.neg(0.1357777), VCon{0.09610699, VCon{F32.neg(0.1423936), VCon{0.07153079, VCon{F32.neg(0.07149822), VCon{F32.neg(0.03733547), VCon{0.2339466, VCon{F32.neg(0.4824447), VCon{F32.neg(0.1440157), VCon{F32.neg(0.7617426), VCon{0.1699532, VCon{F32.neg(0.741998), VCon{F32.neg(0.8018001), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1764959), VCon{F32.neg(0.3167578), VCon{F32.neg(0.5665147), VCon{0.2287756, VCon{0.1389992, VCon{F32.neg(0.3327338), VCon{F32.neg(0.3287029), VCon{0.2085232, VCon{0.1553795, VCon{F32.neg(0.06439619), VCon{F32.neg(0.3174932), VCon{F32.neg(0.4920767), VCon{0.009200075, VCon{0.02309429, VCon{F32.neg(0.1423616), VCon{F32.neg(0.476102), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.5328926, VCon{0.2184003, VCon{F32.neg(0.3636985), VCon{0.1291746, VCon{F32.neg(0.2987558), VCon{0.3008946, VCon{0.07431936, VCon{0.0089153, VCon{0.3927913, VCon{F32.neg(0.4874177), VCon{F32.neg(0.406794), VCon{0.3397993, VCon{0.6332477, VCon{0.0005713459, VCon{0.2272748, VCon{0.09075151, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.4342112), VCon{0.1237967, VCon{F32.neg(0.2102702), VCon{F32.neg(0.06696966), VCon{0.5205461, VCon{0.06329384, VCon{0.7046627, VCon{F32.neg(0.213978), VCon{F32.neg(0.2180045), VCon{F32.neg(0.1299675), VCon{F32.neg(0.593233), VCon{F32.neg(0.03930012), VCon{0.09168245, VCon{F32.neg(0.3079429), VCon{0.2478312, VCon{F32.neg(0.2117955), VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2735747), VCon{F32.neg(0.08166569), VCon{0.08488334, VCon{F32.neg(0.1425688), VCon{F32.neg(0.5706657), VCon{F32.neg(0.5071709), VCon{0.09247071, VCon{0.08140148, VCon{F32.neg(0.3505793), VCon{0.04919387, VCon{F32.neg(0.6047643), VCon{F32.neg(0.2875278), VCon{0.5068336, VCon{F32.neg(0.9329612), VCon{0.4512796, VCon{F32.neg(0.1633161), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1378843), VCon{0.1378765, VCon{F32.neg(0.3007952), VCon{F32.neg(0.02123658), VCon{F32.neg(0.4318266), VCon{F32.neg(0.06120289), VCon{F32.neg(0.0268101), VCon{0.3385141, VCon{F32.neg(0.3372293), VCon{0.02456002, VCon{F32.neg(0.380274), VCon{0.2004462, VCon{0.2511607, VCon{F32.neg(0.4068549), VCon{F32.neg(0.4000576), VCon{0.1475335, VNil{}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.003710765, VCon{0.09067969, VCon{F32.neg(0.4392128), VCon{0.05718864, VCon{0.07773556, VCon{F32.neg(0.2718631), VCon{F32.neg(0.7379695), VCon{F32.neg(0.06636946), VCon{F32.neg(0.109388), VCon{0.2494819, VCon{F32.neg(0.2493618), VCon{0.3228661, VCon{0.5454342, VCon{F32.neg(0.08303321), VCon{0.05069439, VCon{0.3661088, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2737021, VCon{F32.neg(0.50552), VCon{F32.neg(0.3188457), VCon{F32.neg(0.3277087), VCon{0.3328143, VCon{F32.neg(0.1099138), VCon{F32.neg(0.05796146), VCon{F32.neg(0.04138592), VCon{F32.neg(0.09889847), VCon{F32.neg(0.02644644), VCon{F32.neg(0.125301), VCon{F32.neg(0.1824156), VCon{F32.neg(0.1592661), VCon{0.08813371, VCon{F32.neg(0.1243692), VCon{0.06273104, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.281724, VCon{0.0903636, VCon{0.3152438, VCon{F32.neg(0.1761846), VCon{0.1247853, VCon{F32.neg(0.3115683), VCon{F32.neg(0.2788781), VCon{F32.neg(0.3030463), VCon{0.09267554, VCon{F32.neg(0.2029466), VCon{F32.neg(0.00041587), VCon{0.02450592, VCon{0.6354626, VCon{F32.neg(0.03526236), VCon{F32.neg(0.223332), VCon{F32.neg(0.2774378), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.1499707, VCon{F32.neg(0.08159328), VCon{F32.neg(0.2016669), VCon{F32.neg(0.2694742), VCon{0.2636871, VCon{F32.neg(0.1119372), VCon{0.1588001, VCon{0.09450125, VCon{0.1358395, VCon{F32.neg(0.462803), VCon{0.03243819, VCon{0.05818021, VCon{0.5991873, VCon{0.1649781, VCon{0.03119719, VCon{0.3033621, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }
  }

def param_B1() -> Vec:
  VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def param_W2() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.5524087, VCon{F32.neg(0.07704265), VCon{0.6113056, VCon{F32.neg(0.01199642), VCon{0.1253834, VCon{0.4793407, VCon{F32.neg(0.162924), VCon{F32.neg(0.4221659), VCon{0.5062299, VCon{0.1614175, VCon{F32.neg(0.1486985), VCon{0.2368523, VCon{0.03550162, VCon{0.2306259, VCon{F32.neg(0.1362263), VCon{F32.neg(0.07484536), VCon{F32.neg(0.263281), VCon{0.2486186, VCon{F32.neg(0.1666728), VCon{0.6690246, VCon{0.339959, VCon{0.2872391, VCon{0.2257362, VCon{0.23104, VCon{0.2466556, VCon{F32.neg(0.3588727), VCon{0.3652661, VCon{0.3187086, VCon{0.00254522, VCon{0.381507, VCon{F32.neg(0.3721418), VCon{F32.neg(0.06812316), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1038368), VCon{0.06728088, VCon{F32.neg(0.3100696), VCon{0.117332, VCon{F32.neg(0.007032696), VCon{F32.neg(0.0463402), VCon{F32.neg(0.05313379), VCon{F32.neg(0.3316022), VCon{F32.neg(0.4111561), VCon{0.249544, VCon{0.2063746, VCon{F32.neg(0.4415556), VCon{0.2921655, VCon{0.3287193, VCon{F32.neg(0.1442212), VCon{0.02760656, VCon{0.126796, VCon{0.1315067, VCon{F32.neg(0.3065249), VCon{0.1253053, VCon{0.1246222, VCon{F32.neg(0.05514341), VCon{0.5127096, VCon{0.08998221, VCon{F32.neg(0.0674956), VCon{F32.neg(0.1720123), VCon{0.6133336, VCon{F32.neg(0.1198961), VCon{0.09241565, VCon{0.1083199, VCon{F32.neg(0.4546817), VCon{F32.neg(0.5732948), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.5070324, VCon{0.6051009, VCon{0.02020658, VCon{F32.neg(0.2491733), VCon{F32.neg(0.4114571), VCon{0.2734889, VCon{F32.neg(0.1993943), VCon{F32.neg(0.3148899), VCon{F32.neg(0.2884623), VCon{F32.neg(0.04074254), VCon{F32.neg(0.07641879), VCon{0.4921685, VCon{0.2562158, VCon{F32.neg(0.7115679), VCon{F32.neg(0.1944785), VCon{0.1720582, VCon{0.2502722, VCon{F32.neg(0.2893958), VCon{0.1342919, VCon{0.1975517, VCon{F32.neg(0.05209901), VCon{0.9019645, VCon{F32.neg(0.356568), VCon{0.01520162, VCon{0.07077052, VCon{0.2272791, VCon{0.09145211, VCon{F32.neg(0.3108962), VCon{0.4658874, VCon{0.3343865, VCon{F32.neg(0.005776429), VCon{0.1595305, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.6879491, VCon{0.3662294, VCon{F32.neg(0.005184992), VCon{F32.neg(0.3292384), VCon{F32.neg(0.5349636), VCon{F32.neg(0.05057622), VCon{0.4443035, VCon{F32.neg(0.4862733), VCon{F32.neg(0.1302395), VCon{0.1442939, VCon{F32.neg(0.03508006), VCon{F32.neg(0.07165303), VCon{0.2389582, VCon{0.04892366, VCon{0.1765778, VCon{0.2638782, VCon{0.04933221, VCon{F32.neg(0.2470626), VCon{0.2926217, VCon{F32.neg(0.5425027), VCon{F32.neg(0.05824354), VCon{0.05761812, VCon{0.2252611, VCon{0.3236387, VCon{0.2702565, VCon{0.3884147, VCon{0.09938907, VCon{0.2380229, VCon{0.5021718, VCon{0.02542975, VCon{0.04567956, VCon{0.1210155, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.06260186), VCon{F32.neg(0.09261676), VCon{F32.neg(0.2329603), VCon{0.5494964, VCon{0.3158746, VCon{F32.neg(0.4591656), VCon{F32.neg(0.3876711), VCon{0.7416486, VCon{F32.neg(0.1965861), VCon{0.385695, VCon{0.02678014, VCon{F32.neg(0.2752539), VCon{0.0855455, VCon{0.01007876, VCon{0.3456386, VCon{0.2296242, VCon{F32.neg(0.07833178), VCon{F32.neg(0.04092788), VCon{0.1015605, VCon{0.09915575, VCon{0.08230232, VCon{F32.neg(0.2052911), VCon{F32.neg(0.006591011), VCon{0.3131569, VCon{F32.neg(0.9397536), VCon{F32.neg(0.350222), VCon{F32.neg(0.130252), VCon{0.1400661, VCon{F32.neg(0.1023402), VCon{F32.neg(0.4312226), VCon{F32.neg(0.2236629), VCon{F32.neg(0.152376), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.667275, VCon{F32.neg(0.7827028), VCon{0.110725, VCon{0.2073367, VCon{0.1851552, VCon{0.0008565987, VCon{F32.neg(0.1430464), VCon{F32.neg(0.01714958), VCon{F32.neg(0.5930589), VCon{0.2220639, VCon{0.5597804, VCon{0.3418592, VCon{0.2022464, VCon{F32.neg(0.429734), VCon{F32.neg(0.09258365), VCon{0.04369545, VCon{F32.neg(0.1223847), VCon{F32.neg(0.1832798), VCon{F32.neg(0.2288688), VCon{0.008714423, VCon{F32.neg(0.2857882), VCon{F32.neg(0.07042112), VCon{0.02216882, VCon{F32.neg(0.1543165), VCon{0.5128018, VCon{F32.neg(0.5892195), VCon{F32.neg(0.319747), VCon{0.1739811, VCon{F32.neg(0.0103138), VCon{0.04918857, VCon{0.7010346, VCon{0.1744116, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.2518041), VCon{F32.neg(0.03954611), VCon{F32.neg(0.2373551), VCon{0.1918381, VCon{0.00526458, VCon{0.0246105, VCon{F32.neg(0.09783228), VCon{0.07278399, VCon{0.239703, VCon{F32.neg(0.02255953), VCon{0.02505281, VCon{F32.neg(0.09212374), VCon{F32.neg(0.4628977), VCon{F32.neg(0.2680332), VCon{0.1455019, VCon{F32.neg(0.1500552), VCon{F32.neg(0.6074184), VCon{F32.neg(0.1237816), VCon{0.3834138, VCon{F32.neg(0.2305551), VCon{0.4156646, VCon{0.3618862, VCon{0.2173694, VCon{0.2855297, VCon{F32.neg(0.4941878), VCon{0.4152342, VCon{F32.neg(0.2241814), VCon{0.2996086, VCon{0.07876677, VCon{F32.neg(0.3740821), VCon{0.392187, VCon{F32.neg(0.2593752), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.2762368, VCon{F32.neg(0.233395), VCon{0.1595055, VCon{F32.neg(0.0674637), VCon{0.1117916, VCon{0.3150979, VCon{0.3785255, VCon{F32.neg(0.1259821), VCon{F32.neg(0.09649511), VCon{0.1681244, VCon{F32.neg(0.1385981), VCon{F32.neg(0.05001277), VCon{0.1047497, VCon{0.05986796, VCon{0.1118944, VCon{0.09618146, VCon{0.02440571, VCon{0.8930708, VCon{0.5548355, VCon{0.3834581, VCon{F32.neg(0.9250847), VCon{F32.neg(0.0965131), VCon{F32.neg(0.2536466), VCon{F32.neg(0.1154425), VCon{F32.neg(0.015154), VCon{F32.neg(0.1999677), VCon{F32.neg(0.5241979), VCon{0.1689789, VCon{0.07097583, VCon{F32.neg(0.236786), VCon{0.07577267, VCon{0.4492018, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{0.1248401, VCon{0.2221126, VCon{0.1433058, VCon{0.3959259, VCon{0.4152714, VCon{F32.neg(0.01202948), VCon{0.6208829, VCon{0.117778, VCon{0.283781, VCon{F32.neg(0.08555058), VCon{0.3341562, VCon{0.08362812, VCon{F32.neg(0.06256562), VCon{0.4796002, VCon{0.2016215, VCon{F32.neg(0.07339706), VCon{F32.neg(0.1641158), VCon{F32.neg(0.5038757), VCon{F32.neg(0.2585794), VCon{0.0712879, VCon{0.1207959, VCon{0.0608828, VCon{F32.neg(0.1835584), VCon{F32.neg(0.0798724), VCon{F32.neg(0.4381834), VCon{0.2336145, VCon{F32.neg(0.535372), VCon{F32.neg(0.1308403), VCon{F32.neg(0.4197761), VCon{F32.neg(0.1803914), VCon{0.3025002, VCon{F32.neg(0.1808247), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.2347941), VCon{F32.neg(0.6917489), VCon{0.009847756, VCon{F32.neg(0.4115978), VCon{F32.neg(0.351348), VCon{0.4331595, VCon{0.4306625, VCon{0.6821354, VCon{0.1531181, VCon{0.04369166, VCon{F32.neg(0.05063013), VCon{0.4632779, VCon{0.2602096, VCon{F32.neg(0.05403543), VCon{F32.neg(0.0877725), VCon{0.387255, VCon{0.5019457, VCon{F32.neg(0.02914057), VCon{F32.neg(0.1515588), VCon{F32.neg(0.08039165), VCon{0.01845334, VCon{F32.neg(0.2699187), VCon{0.2155953, VCon{0.6849329, VCon{0.2001923, VCon{0.09141108, VCon{0.1759753, VCon{0.3032071, VCon{0.04529113, VCon{0.125228, VCon{0.07994404, VCon{F32.neg(0.2479219), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.3145131, VCon{F32.neg(0.4382001), VCon{F32.neg(0.5952042), VCon{F32.neg(0.1476175), VCon{0.03747368, VCon{0.06137336, VCon{F32.neg(0.296261), VCon{F32.neg(0.1237308), VCon{0.2204112, VCon{F32.neg(0.2926603), VCon{F32.neg(0.3961423), VCon{F32.neg(0.1058877), VCon{0.4706522, VCon{F32.neg(0.06252909), VCon{0.3387682, VCon{0.01026143, VCon{F32.neg(0.7055138), VCon{F32.neg(0.04515592), VCon{F32.neg(0.7907149), VCon{0.05531826, VCon{F32.neg(0.5885589), VCon{0.4201517, VCon{0.111628, VCon{0.1280307, VCon{0.5380601, VCon{0.0446212, VCon{0.1651604, VCon{0.3813629, VCon{F32.neg(0.176244), VCon{0.008048472, VCon{0.3667053, VCon{F32.neg(0.1341535), VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4105935, VCon{F32.neg(0.1339359), VCon{F32.neg(0.3191136), VCon{0.1302675, VCon{F32.neg(0.3631621), VCon{0.1996851, VCon{F32.neg(0.02022804), VCon{F32.neg(0.2742178), VCon{0.1803962, VCon{0.09487563, VCon{F32.neg(0.02853857), VCon{F32.neg(0.3714866), VCon{F32.neg(0.07271692), VCon{F32.neg(0.343713), VCon{0.5162278, VCon{0.5156993, VCon{F32.neg(0.8118374), VCon{F32.neg(0.1942708), VCon{F32.neg(0.3073932), VCon{0.146581, VCon{0.1769854, VCon{F32.neg(0.3245609), VCon{0.5824347, VCon{F32.neg(0.1534103), VCon{F32.neg(0.2876326), VCon{F32.neg(0.3426887), VCon{F32.neg(0.0761322), VCon{0.03826365, VCon{F32.neg(0.1912719), VCon{F32.neg(0.7045162), VCon{0.1126679, VCon{0.1933709, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{0.05463835, VCon{F32.neg(0.142232), VCon{F32.neg(0.0358237), VCon{F32.neg(0.371141), VCon{0.1874992, VCon{F32.neg(0.1321576), VCon{F32.neg(0.9810279), VCon{0.2551906, VCon{F32.neg(0.3800038), VCon{F32.neg(0.1424945), VCon{F32.neg(0.2421443), VCon{0.2221977, VCon{F32.neg(0.2366295), VCon{F32.neg(0.5409209), VCon{0.1396331, VCon{F32.neg(0.1497583), VCon{F32.neg(0.400243), VCon{0.4347078, VCon{F32.neg(0.0872852), VCon{F32.neg(0.3246936), VCon{F32.neg(0.4749334), VCon{0.1146546, VCon{0.005268193, VCon{F32.neg(0.01735197), VCon{0.2453761, VCon{F32.neg(0.1345567), VCon{0.718644, VCon{0.5332385, VCon{F32.neg(0.1106565), VCon{0.2005257, VCon{F32.neg(0.3003105), VCon{0.0183919, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.4936557), VCon{0.1459575, VCon{0.2821327, VCon{0.06773447, VCon{0.237199, VCon{F32.neg(0.2547078), VCon{0.0357459, VCon{0.2638862, VCon{F32.neg(0.3817814), VCon{0.2988833, VCon{0.05383453, VCon{F32.neg(0.5259936), VCon{0.6751171, VCon{0.4048487, VCon{0.4248991, VCon{F32.neg(0.7231301), VCon{F32.neg(0.169505), VCon{F32.neg(0.2542908), VCon{0.06680335, VCon{F32.neg(0.3894747), VCon{F32.neg(0.5971744), VCon{0.03426515, VCon{0.1615756, VCon{0.2405491, VCon{F32.neg(0.07225688), VCon{F32.neg(0.1479333), VCon{0.3452997, VCon{0.63387, VCon{F32.neg(0.2402893), VCon{F32.neg(0.1349427), VCon{F32.neg(0.02495426), VCon{0.2263541, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{0.2486113, VCon{F32.neg(0.07900777), VCon{0.09634876, VCon{F32.neg(0.481191), VCon{0.2643984, VCon{F32.neg(0.6112388), VCon{F32.neg(0.2024364), VCon{F32.neg(0.01767862), VCon{F32.neg(0.147922), VCon{F32.neg(0.07490626), VCon{0.4008108, VCon{F32.neg(0.1494823), VCon{0.06847779, VCon{F32.neg(0.198691), VCon{0.1736124, VCon{F32.neg(0.5479786), VCon{F32.neg(0.05275464), VCon{0.3291526, VCon{F32.neg(0.3067451), VCon{F32.neg(0.3551648), VCon{0.24599, VCon{F32.neg(0.2120048), VCon{0.09651199, VCon{0.288924, VCon{F32.neg(0.4278587), VCon{F32.neg(0.2252433), VCon{0.3474106, VCon{0.2708139, VCon{0.03226212, VCon{F32.neg(0.2230817), VCon{F32.neg(0.2269878), VCon{0.1257796, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.253954), VCon{F32.neg(0.4008758), VCon{0.3806768, VCon{0.1036052, VCon{F32.neg(0.2400555), VCon{0.2645022, VCon{F32.neg(0.5074211), VCon{F32.neg(0.1535497), VCon{0.4404583, VCon{0.3192959, VCon{F32.neg(0.2380251), VCon{F32.neg(0.1351155), VCon{F32.neg(0.06419134), VCon{0.04766367, VCon{0.06182503, VCon{0.5195549, VCon{0.1547285, VCon{0.2016653, VCon{F32.neg(0.1340454), VCon{0.3913986, VCon{F32.neg(0.2675014), VCon{0.03230365, VCon{F32.neg(0.4208505), VCon{F32.neg(0.225902), VCon{F32.neg(0.1681868), VCon{0.06790732, VCon{F32.neg(0.5030163), VCon{0.140976, VCon{F32.neg(0.06487523), VCon{F32.neg(0.07930939), VCon{F32.neg(0.002935274), VCon{0.3499662, VNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_B2() -> Vec:
  VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}}}}}}}

def param_W3() -> MatTree:
  MTNode{
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.3809308), VCon{0.4103262, VCon{0.3053726, VCon{0.04176147, VCon{F32.neg(0.1601893), VCon{0.4271119, VCon{F32.neg(0.3577522), VCon{0.01473633, VCon{F32.neg(0.1490645), VCon{F32.neg(0.194446), VCon{F32.neg(0.03434518), VCon{F32.neg(0.5487165), VCon{0.6885239, VCon{0.1760808, VCon{0.2927736, VCon{F32.neg(0.334051), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.4217006), VCon{F32.neg(0.1158591), VCon{F32.neg(0.0004729992), VCon{F32.neg(0.4497558), VCon{0.4652205, VCon{0.3533668, VCon{F32.neg(0.3292023), VCon{0.3374027, VCon{F32.neg(0.2562741), VCon{F32.neg(0.1432509), VCon{0.2902475, VCon{0.2699359, VCon{0.2894587, VCon{F32.neg(0.06560286), VCon{F32.neg(0.008421654), VCon{0.5143076, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.1309229), VCon{F32.neg(0.142174), VCon{0.1845181, VCon{0.002646405, VCon{F32.neg(0.1000825), VCon{F32.neg(0.169318), VCon{0.1186925, VCon{0.540545, VCon{F32.neg(0.04598203), VCon{0.119396, VCon{0.09207059, VCon{F32.neg(0.01657573), VCon{F32.neg(0.2320113), VCon{F32.neg(0.3157556), VCon{0.2689338, VCon{0.6071158, VNil{}}}}}}}}}}}}}}}}}},
    MTNode{
    MTLeaf{VCon{F32.neg(0.7226905), VCon{0.3044546, VCon{F32.neg(0.2707257), VCon{0.4399792, VCon{0.002195143, VCon{F32.neg(0.05310226), VCon{0.1594151, VCon{0.3108394, VCon{0.1562454, VCon{0.04418903, VCon{F32.neg(0.3951918), VCon{0.273138, VCon{F32.neg(0.5263999), VCon{F32.neg(0.1125531), VCon{0.8429411, VCon{F32.neg(0.1090321), VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.4541223, VCon{F32.neg(0.4811622), VCon{0.02672597, VCon{F32.neg(0.3089521), VCon{F32.neg(0.7882416), VCon{F32.neg(0.3909368), VCon{0.397837, VCon{0.2445249, VCon{F32.neg(0.2431529), VCon{F32.neg(0.1842215), VCon{F32.neg(0.2057622), VCon{F32.neg(0.03955074), VCon{F32.neg(0.1929068), VCon{0.004705095, VCon{0.2867329, VCon{F32.neg(0.2488189), VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  },
    MTNode{
    MTNode{
    MTLeaf{VCon{F32.neg(0.1413697), VCon{F32.neg(0.05853821), VCon{0.0917004, VCon{0.2418899, VCon{F32.neg(0.1094536), VCon{0.1391219, VCon{0.3000479, VCon{F32.neg(0.849409), VCon{0.7094266, VCon{0.1658011, VCon{F32.neg(0.2131201), VCon{0.2674486, VCon{F32.neg(0.4978338), VCon{F32.neg(0.2612907), VCon{0.03638417, VCon{0.06047739, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{F32.neg(0.1179084), VCon{F32.neg(0.4734731), VCon{F32.neg(0.1535368), VCon{F32.neg(0.3306589), VCon{0.1456496, VCon{F32.neg(0.4534072), VCon{F32.neg(0.1743146), VCon{0.08610512, VCon{F32.neg(0.9529553), VCon{F32.neg(0.7250273), VCon{F32.neg(0.3127199), VCon{F32.neg(0.2790585), VCon{0.1736078, VCon{F32.neg(0.2177355), VCon{F32.neg(0.06370622), VCon{0.7134178, VNil{}}}}}}}}}}}}}}}}}}
  },
    MTNode{
    MTLeaf{VCon{F32.neg(0.01629202), VCon{0.4005559, VCon{0.1479322, VCon{0.7736838, VCon{0.04635219, VCon{0.02374435, VCon{0.1145414, VCon{F32.neg(0.4183912), VCon{F32.neg(0.4607526), VCon{F32.neg(0.09829436), VCon{0.1095869, VCon{F32.neg(0.1391511), VCon{F32.neg(0.1307275), VCon{0.1768148, VCon{F32.neg(0.3106896), VCon{0.2082777, VNil{}}}}}}}}}}}}}}}}}},
    MTNode{
    MTLeaf{VCon{F32.neg(0.5540205), VCon{F32.neg(0.4097742), VCon{F32.neg(0.08164238), VCon{0.656444, VCon{0.01500337, VCon{0.07257716, VCon{0.02282073, VCon{0.3544966, VCon{F32.neg(0.2884977), VCon{F32.neg(0.172344), VCon{0.0733152, VCon{F32.neg(0.08919846), VCon{0.3064471, VCon{F32.neg(0.1292298), VCon{F32.neg(0.1622672), VCon{0.0005404382, VNil{}}}}}}}}}}}}}}}}}},
    MTLeaf{VCon{0.185997, VCon{F32.neg(0.1409323), VCon{0.5763251, VCon{F32.neg(0.2344698), VCon{F32.neg(0.08015345), VCon{F32.neg(0.4552215), VCon{0.616864, VCon{F32.neg(0.1502051), VCon{F32.neg(0.1210838), VCon{F32.neg(0.05028835), VCon{F32.neg(0.04284434), VCon{F32.neg(0.4111427), VCon{F32.neg(0.02445354), VCon{0.7104704, VCon{F32.neg(0.5019969), VCon{0.2856036, VNil{}}}}}}}}}}}}}}}}}}
  }
  }
  }
  }

def param_B3() -> Vec:
  VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VCon{0.0, VNil{}}}}}}}}}}}

# === Forward Inference Computation Graph ===
def forward(+input: Vec) -> Vec:
  t0 : Vec = linear_layer(input, param_W1(), param_B1())
  t1 : Vec = vec_relu(t0)
  t2 : Vec = linear_layer(t1, param_W2(), param_B2())
  t3 : Vec = vec_relu(t2)
  t4 : Vec = linear_layer(t3, param_W3(), param_B3())
  t5 : Vec = vec_softmax(t4)
  t5

# === Main Entrypoint & GPU Parallel Verification ===
def main() -> IO(Unit):
  do IO<Unit>:
    test_x : Vec = VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VCon{0.5, VNil{}}}}}}}}}}}}}}}}}
    
    # Run forward inference with parallel GPU tree dispatch:
    +out : Vec = forward!(test_x)
    
    predicted_class : U32 = vec_argmax(out)
    
    IO.print("BEND_ONNX_OUTPUT: " ++ vec_show(out))
    IO.print("BEND_PREDICTED_CLASS: " ++ U32.show(predicted_class))
