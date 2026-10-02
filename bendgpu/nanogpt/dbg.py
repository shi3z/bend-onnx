import os, sys, subprocess, numpy as np
sys.argv=[sys.argv[0]]
import run as R
b=R.b; torch=b.torch
flat,pf=R.params()
exe=R.build(4)
env=dict(os.environ, BG_DUMP="/tmp/bend_train/state1.bin")
print(subprocess.run([exe,"1","12289",pf,"verify"],capture_output=True,text=True,env=env).stdout)
st=np.fromfile("/tmp/bend_train/state1.bin",np.float32)
print("nan:",np.isnan(st).sum(), "step slot", st[12288], "loss", st[12289])
m_=b.make_model(16); b.load_flat(m_,16,flat); m_.train()
xs=torch.tensor([b.X[i%4] for i in range(4)]); ys=torch.tensor([b.Y[i%4] for i in range(4)])
_,loss=m_(xs,ys); loss.backward()
for p in m_.parameters(): p.data=p.grad.clone() if p.grad is not None else p.data*0
g=b.flat_params(m_,16)
mm=st[4096:4096+4048]   # m = (1-b1)*g_bend
gb=mm/0.1
names=[("wte",0,512),("wpe",512,736),("ln1g",736,752),("ln1b",752,768),("wq",768,1024),("bq",1024,1040),("wk",1040,1296),("bk",1296,1312),("wv",1312,1568),("bv",1568,1584),("wo",1584,1840),("bo",1840,1856),("ln2g",1856,1872),("ln2b",1872,1888),("wfc",1888,2912),("bfc",2912,2976),("wmp",2976,4000),("bmp",4000,4016),("lnfg",4016,4032),("lnfb",4032,4048)]
for n,a,e in names:
    d=np.abs(gb[a:e]-g[a:e]).max(); sc=np.abs(g[a:e]).max()
    print(f"{n:5s} max|g|={sc:.4e} maxdiff={d:.3e}")
print("ln1g bend", gb[736:742], "torch", g[736:742])
print("wo bend", gb[1584:1590], "torch", g[1584:1590])
print("wte bend", gb[0:6], "torch", g[0:6])
print("m region sample", st[4096+736:4096+740], "v", st[8192+736:8192+740], "p", st[736:740])
