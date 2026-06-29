import sys, cupy as cp
import run_adl_queries as rq
q=int(sys.argv[1]); fp=sys.argv[2]
fn=getattr(rq,f"query{q}_gpu")
mp=cp.get_default_memory_pool()
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    fn(fp)                      # warm (JIT)
mp.free_all_blocks(); 
# can't easily reset peak across cupy versions; use device free delta instead
free0,total=cp.cuda.Device(0).mem_info
with contextlib.redirect_stdout(io.StringIO()):
    fn(fp)
free1,_=cp.cuda.Device(0).mem_info
peak_pool=mp.total_bytes()
print(f"Q{q}: pool_peak={peak_pool/1e9:.2f} GB  dev_free_drop={(free0-free1)/1e9:.2f} GB  dev_total={total/1e9:.1f} GB")
