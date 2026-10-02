import os, sys, subprocess
B=int(sys.argv[1]) if len(sys.argv)>1 else 4096
sys.argv=[sys.argv[0]]
import run as R
flat,pf=R.params()
exe=R.build(B)
print(subprocess.run([exe,"20","12289",pf],capture_output=True,text=True,env=dict(os.environ,BG_PROF="1")).stdout)
