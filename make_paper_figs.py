"""Light-theme figures for the GPU-Accelerated Awkward Arrays paper.

- layout.png          : Awkward (offsets, content) ragged layout  (Background)
- akmin_passes.png    : hand-written ak.min multi-pass + global-memory round trips (Background)
- architecture.png    : new backend — hand-written CUDA C++ → pure Python / cuda.compute
- loc_comparison.png  : LOC reduction
- benchmark_speedup.png: ADL speedup vs event count (real data)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/home/coder/scipy_proceedings/papers/awkward_cuda_compute"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

BLUE="#1f6fb2"; GREEN="#4f9d2a"; GREY="#8a9097"; PURPLE="#7b5ea7"; INK="#1a1a1a"; RED="#b5402f"
LGREEN="#e3f0d4"; LBLUE="#d7e6f2"; LGREY="#ececef"; LPURP="#ece4f4"; LORANGE="#fbe6cf"

def box(ax,x,y,w,h,txt,face,edge,fs=10,dashed=False,bold=False,sub=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.6,edgecolor=edge,facecolor=face,
                 linestyle="--" if dashed else "-",zorder=2))
    if sub:
        ax.text(x+w/2,y+h*0.64,txt,ha="center",va="center",fontsize=fs,color=INK,
                fontweight="bold" if bold else "normal",zorder=3)
        ax.text(x+w/2,y+h*0.28,sub,ha="center",va="center",fontsize=8.0,color="#555",zorder=3)
    else:
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=INK,
                fontweight="bold" if bold else "normal",zorder=3)

def arrow(ax,p1,p2,color=INK,lw=1.7,dashed=False,style="-|>"):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=14,lw=lw,color=color,
                 linestyle="--" if dashed else "-",zorder=1,shrinkA=3,shrinkB=3))

# ----------------------------------------------------------------- layout
def fig_layout():
    fig,ax=plt.subplots(figsize=(8.4,3.7)); ax.set_xlim(0,12); ax.set_ylim(0,5.2); ax.axis("off")
    vals=["1.1","2.2","3.3","4.4","5.5"]
    seg_color=[LBLUE,LBLUE,LBLUE,LGREEN,LGREEN]; seg_edge=[BLUE,BLUE,BLUE,GREEN,GREEN]
    # logical view
    ax.text(0.1,4.7,"logical (ragged) view:",fontsize=9.5,color="#555")
    ax.text(0.1,4.2,"[[1.1, 2.2, 3.3],  [],  [4.4, 5.5]]",fontsize=12,family="monospace",color=INK)
    # content buffer
    x0=2.2; cw=1.25; y=2.2
    ax.text(0.1,y+cw/2,"content",fontsize=10,ha="left",va="center",fontweight="bold")
    for i,v in enumerate(vals):
        ax.add_patch(Rectangle((x0+i*cw,y),cw,0.85,facecolor=seg_color[i],edgecolor=seg_edge[i],lw=1.5))
        ax.text(x0+i*cw+cw/2,y+0.42,v,ha="center",va="center",fontsize=11,family="monospace")
        ax.text(x0+i*cw+cw/2,y-0.28,str(i),ha="center",va="center",fontsize=8,color="#888")
    # offsets
    yo=0.35
    ax.text(0.1,yo+0.3,"offsets",fontsize=10,ha="left",va="center",fontweight="bold")
    offs=[0,3,3,5]; ox0=2.2; ow=1.0
    for i,o in enumerate(offs):
        ax.add_patch(Rectangle((ox0+i*ow,yo),ow,0.6,facecolor=LGREY,edgecolor=GREY,lw=1.3))
        ax.text(ox0+i*ow+ow/2,yo+0.3,str(o),ha="center",va="center",fontsize=11,family="monospace")
    # boundary arrows offsets -> content positions
    for o in [0,3,5]:
        ax.add_patch(FancyArrowPatch((ox0+(offs.index(o))*ow+ow/2, yo+0.62),
                     (x0+o*cw, y-0.05),arrowstyle="-",lw=1.0,color="#aaa",
                     connectionstyle="arc3,rad=0.0",zorder=0))
    # braces for lists
    for (a,b,col,lbl) in [(0,3,BLUE,"list 0"),(3,5,GREEN,"list 2")]:
        ax.plot([x0+a*cw+0.05,x0+b*cw-0.05],[y+1.0,y+1.0],color=col,lw=2)
        ax.text(x0+(a+b)/2*cw,y+1.25,lbl,ha="center",fontsize=8.5,color=col)
    ax.text(x0+3*cw,y+1.05,"list 1 = [] (empty)",ha="center",fontsize=8.0,color="#999")
    ax.set_title("Awkward ragged layout: variable-length lists as flat content + offsets",
                 fontsize=11.5,fontweight="bold")
    fig.savefig(f"{OUT}/layout.png"); plt.close(fig); print("wrote layout.png")

# ----------------------------------------------------------------- ak.min passes
def fig_akmin():
    fig,ax=plt.subplots(figsize=(9.0,4.4)); ax.set_xlim(0,12); ax.set_ylim(0,6.7); ax.axis("off")
    ky=3.9; kw=3.2; kh=1.5
    xs=[0.5,4.4,8.3]
    titles=[("Kernel 1","initialize scratch\nto identity"),
            ("Kernel 2","block reduce\n(shared mem, __syncthreads)"),
            ("Kernel 3","copy scratch\n→ output")]
    for x,(t,s) in zip(xs,titles):
        ax.add_patch(FancyBboxPatch((x,ky),kw,kh,boxstyle="round,pad=0.02,rounding_size=0.05",
                     linewidth=1.6,edgecolor=GREY,facecolor=LGREY,zorder=2))
        ax.text(x+kw/2,ky+kh*0.74,t,ha="center",va="center",fontsize=10.5,fontweight="bold",color=INK,zorder=3)
        ax.text(x+kw/2,ky+kh*0.32,s,ha="center",va="center",fontsize=8.2,color="#555",zorder=3)
    # global memory band
    gy=1.1; ax.add_patch(Rectangle((0.5,gy),11.0,0.95,facecolor=LORANGE,edgecolor=RED,lw=1.5))
    ax.text(6.0,gy+0.48,"GPU global memory  (scratch buffer)",ha="center",va="center",
            fontsize=10.5,color=RED,fontweight="bold")
    # write/read round trips
    for x in xs:
        cx=x+kw/2
        arrow(ax,(cx-0.4,ky),(cx-0.4,gy+0.95),color=RED,lw=1.6)          # write down
    for x in xs[1:]:
        cx=x+kw/2
        arrow(ax,(cx+0.4,gy+0.95),(cx+0.4,ky),color=RED,lw=1.6)          # read up
    ax.text(xs[0]+kw/2-0.9,(ky+gy+0.95)/2+0.15,"write",fontsize=8,color=RED,ha="right")
    ax.text(xs[1]+kw/2+0.9,(ky+gy+0.95)/2+0.15,"read",fontsize=8,color=RED,ha="left")
    # launch/sync between kernels
    for x in [4.0,7.9]:
        ax.plot([x,x],[ky-0.25,ky+kh+0.25],ls=(0,(2,2)),color="#999",lw=1.3)
        ax.text(x,ky+kh+0.45,"launch\n+ sync",ha="center",va="bottom",fontsize=7.5,color="#777")
    ax.annotate("",xy=(4.4,ky+kh/2),xytext=(3.7,ky+kh/2),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.5))
    ax.annotate("",xy=(8.3,ky+kh/2),xytext=(7.6,ky+kh/2),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.5))
    ax.set_title("Hand-written ak.min: three kernel launches, each round-tripping through global memory",
                 fontsize=11,fontweight="bold")
    fig.savefig(f"{OUT}/akmin_passes.png"); plt.close(fig); print("wrote akmin_passes.png")

# ----------------------------------------------------------------- architecture
def fig_architecture():
    # Dispatch + compile flow: the dispatcher routes GPU work to the pure-Python
    # cuda.compute backend, where Awkward's ops/iterators and cuda.compute's
    # CUB/Thrust kernels are each JIT-compiled to LTO-IR, then JIT-linked into a
    # single fused CUDA kernel.
    fig,ax=plt.subplots(figsize=(13.2,8.6)); ax.set_xlim(0,15); ax.set_ylim(0,10); ax.axis("off")

    def abox(cx,cy,w,h,face,edge,title,sub=None,bold=True,dashed=False,
             title_size=15,sub_size=11):
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
                     boxstyle="round,pad=0.02,rounding_size=0.18",linewidth=2.0,
                     facecolor=face,edgecolor=edge,linestyle="--" if dashed else "-",
                     mutation_aspect=1.0,zorder=2))
        if sub:
            ax.text(cx,cy+0.16,title,ha="center",va="center",fontsize=title_size,
                    fontweight="bold" if bold else "normal",color=INK,zorder=3)
            ax.text(cx,cy-0.27,sub,ha="center",va="center",fontsize=sub_size,
                    color="#555",zorder=3)
        else:
            ax.text(cx,cy,title,ha="center",va="center",fontsize=title_size,
                    fontweight="bold" if bold else "normal",color=INK,zorder=3)

    def aarrow(x0,y0,x1,y1,color=INK):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",
                     mutation_scale=20,linewidth=2.2,color=color,shrinkA=0,shrinkB=0,zorder=1))

    ax.text(7.5,9.6,"New backend: hand-written CUDA C++ → pure Python via cuda.compute",
            ha="center",va="center",fontsize=17,fontweight="bold",color=INK)

    # user API + dispatcher
    abox(7.5,8.6,4.6,1.0,LBLUE,BLUE,"ak.Array — Python API","unchanged: ak.min, ak.sort, …")
    abox(7.5,7.1,4.6,0.9,LGREY,GREY,"Awkward dispatcher",bold=False)

    # branch row
    cpu_x,hw_x,cc_x=2.4,7.0,11.6
    abox(cpu_x,5.5,3.6,1.0,LGREY,GREY,"CPU kernels","C++",bold=False)
    abox(hw_x,5.5,3.8,1.0,LGREY,GREY,"hand-written CUDA","legacy GPU, CUDA C++",
         bold=False,dashed=True)
    abox(cc_x,5.5,4.0,1.0,LGREEN,GREEN,"cuda.compute backend","new GPU, pure Python")
    aarrow(7.0,6.65,cpu_x+0.9,6.0); aarrow(7.0,6.65,hw_x,6.0)
    aarrow(7.0,6.65,cc_x-0.9,6.0,color=GREEN)

    # two parallel JIT compiles to LTO-IR
    ops_x,ker_x,comp_y=10.0,13.2,3.7
    ax.text(ops_x,4.55,"Awkward backend",ha="center",va="center",fontsize=10,
            style="italic",color=GREEN)
    ax.text(ker_x,4.55,"cuda.compute library",ha="center",va="center",fontsize=10,
            style="italic",color=GREEN)
    abox(ops_x,comp_y,3.0,1.0,LGREEN,GREEN,"ops + iterators","JIT compiled → LTO-IR",
         bold=False,title_size=12.5,sub_size=10)
    abox(ker_x,comp_y,3.0,1.0,LGREEN,GREEN,"CUB/Thrust kernels","JIT compiled → LTO-IR",
         bold=False,title_size=12.5,sub_size=10)
    aarrow(cc_x-0.6,5.0,ops_x,comp_y+0.5,color=GREEN)
    aarrow(cc_x+0.6,5.0,ker_x,comp_y+0.5,color=GREEN)

    # JIT link -> fused kernel -> GPU
    link_y=2.0
    abox(11.6,link_y,4.8,1.0,LGREEN,GREEN,"JIT link","fused CUDA kernel",
         title_size=14,sub_size=11)
    aarrow(ops_x,comp_y-0.5,10.6,link_y+0.5,color=GREEN)
    aarrow(ker_x,comp_y-0.5,12.6,link_y+0.5,color=GREEN)
    abox(11.6,0.6,4.0,0.85,LGREEN,GREEN,"GPU execution",bold=False,title_size=14)
    aarrow(11.6,link_y-0.5,11.6,1.03,color=GREEN)

    fig.savefig(f"{OUT}/architecture.png"); plt.close(fig); print("wrote architecture.png")

# ----------------------------------------------------------------- loc
def fig_loc():
    # Measured, counting only maintained code (excludes .cu kept solely as fallback):
    #   Awkward 2.8.11: 8,288 lines CUDA C++ + 317 lines Python glue.
    #   Latest:         2,736 lines CUDA C++ (53 unmigrated kernels) + 3,996 lines Python.
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    x=[0,1]; w=0.55
    cpp=[8288,2736]; py=[317,3996]
    ax.bar(x,cpp,w,color=GREY,edgecolor=INK,linewidth=0.9,label="CUDA C++")
    ax.bar(x,py,w,bottom=cpp,color=GREEN,edgecolor=INK,linewidth=0.9,label="Python")
    for xi,c,p in zip(x,cpp,py):
        if c>250: ax.text(xi,c/2,f"{c:,}",ha="center",va="center",color="white",fontsize=10,fontweight="bold")
        if p>250: ax.text(xi,c+p/2,f"{p:,}",ha="center",va="center",color="white",fontsize=10,fontweight="bold")
        ax.text(xi,c+p+140,f"{c+p:,}",ha="center",fontsize=10.5,fontweight="bold",color=INK)
    ax.annotate("",xy=(1,2736),xytext=(0.0,8288),
                arrowprops=dict(arrowstyle="->",color=GREY,lw=1.4,ls="--"))
    ax.text(0.5,6900,"C++ to maintain\n−67%",color="#555",fontsize=9.5,ha="center",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Awkward 2.8\n(before cuda.compute)","Awkward latest\n(cuda.compute)"],fontsize=10)
    ax.set_ylabel("Lines of GPU kernel code (maintained)")
    ax.set_ylim(0,10000); ax.set_xlim(-0.6,1.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper right",fontsize=10,title="language")
    ax.set_title("GPU kernel code: less C++ to maintain, mostly Python",fontsize=11.5,fontweight="bold")
    fig.savefig(f"{OUT}/loc_comparison.png"); plt.close(fig); print("wrote loc_comparison.png")

# ----------------------------------------------------------------- benchmark
def fig_benchmark():
    base="/home/coder/columnar_gpu_bench/columnar_gpu/logs"
    sizes=[("100k",1e5),("1M",1e6),("10M",1e7)]; data={}
    for tag,N in sizes:
        rows={}
        for ln in open(f"{base}/cmp_{tag}.txt"):
            ln=ln.strip()
            if ln.startswith("{"):
                r=json.loads(ln); rows.setdefault(r["q"],{})[r["label"]]=r
        for q,by in rows.items():
            b=by.get("baseline"); a=by.get("awkward3"); sp=None
            if b and a and b.get("ok") and a.get("ok") and a["comp"]>1e-4: sp=b["comp"]/a["comp"]
            data.setdefault(q,[]).append((N,sp))
    fig,ax=plt.subplots(figsize=(7.0,4.6))
    series={5:("Q5  di-muon mass",BLUE,"o"),6:("Q6  trijet",GREEN,"s"),
            4:("Q4  MET, ≥2 jets",PURPLE,"^"),7:("Q7  lepton-jet HT",GREY,"D")}
    for q,(lab,col,mk) in series.items():
        pts=[(N,sp) for N,sp in data.get(q,[]) if sp is not None]
        if not pts: continue
        ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=mk,color=col,lw=2,ms=7,label=lab,zorder=3)
        for x,y in pts:
            if y>=2:
                # nudge right-edge (10M) labels leftward so they stay on-frame
                dx=-24 if x>=9e6 else 6
                ax.annotate(f"{y:.0f}×",(x,y),textcoords="offset points",xytext=(dx,4),
                            fontsize=9,color=col,fontweight="bold",clip_on=False)
    ax.axhline(1.0,color=RED,lw=1.2,ls="--",zorder=1)
    ax.text(1.12e5,1.06,"parity",color=RED,fontsize=8.5,va="bottom",clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1e5,1e6,1e7]); ax.set_xticklabels(["100k","1M","10M"])
    ax.set_yticks([0.5,1,2,5,10,20,50,100]); ax.set_yticklabels(["0.5","1","2","5","10","20","50","100"])
    ax.set_xlim(7e4,1.5e7); ax.set_ylim(0.45,170)
    ax.set_xlabel("Number of events"); ax.set_ylabel("Speedup  (hand-written ÷ cuda.compute)")
    ax.set_title("ADL benchmarks: cuda.compute speedup grows with scale",fontsize=11.5,fontweight="bold")
    ax.grid(True,which="both",ls=":",alpha=0.4); ax.legend(frameon=False,fontsize=9,loc="upper left")
    ax.text(0.99,0.02,"Q3 & Q8: hand-written backend fails; cuda.compute succeeds",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=8,style="italic",color="#444")
    fig.savefig(f"{OUT}/benchmark_speedup.png"); plt.close(fig); print("wrote benchmark_speedup.png")

if __name__=="__main__":
    fig_layout(); fig_akmin(); fig_architecture(); fig_loc(); fig_benchmark()
