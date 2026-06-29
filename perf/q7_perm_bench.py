"""Q7 compute stage: array-materialized dR (current query7c) vs PermutationIterator-fused
(zero float temporaries). Same pair-index build; only the dR + HT reductions differ.
Times the compute stage warm at 1M and 10M; checks the two agree."""
import sys, os, time, statistics
from math import sqrt, pi as MPI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "columnar_gpu"))
os.environ.setdefault("AK_BENCH_READER", "cudf")
import numpy as np, cupy as cp, awkward as ak, run_adl_queries as rq
from cuda.compute import segmented_reduce, OpKind
from cuda.compute.iterators import PermutationIterator, ZipIterator, TransformIterator

def jb(arr):
    lay = arr.layout
    return cp.asarray(lay.offsets.data).astype(cp.int64), cp.ascontiguousarray(cp.asarray(lay.content.data))

def load(fp):
    t = rq.cudf.read_parquet(fp, columns=["Jet_pt","Jet_eta","Jet_phi","Electron_eta","Electron_phi","Muon_eta","Muon_phi"])
    g = rq.cudf_to_awkward
    off_j, pt_j = jb(g(t["Jet_pt"])); pt_j=pt_j.astype(cp.float32)
    _, eta_j = jb(g(t["Jet_eta"])); eta_j=eta_j.astype(cp.float32)
    _, phi_j = jb(g(t["Jet_phi"])); phi_j=phi_j.astype(cp.float32)
    le = ak.to_packed(ak.concatenate([g(t["Electron_eta"]), g(t["Muon_eta"])], axis=1))
    lp = ak.to_packed(ak.concatenate([g(t["Electron_phi"]), g(t["Muon_phi"])], axis=1))
    off_l, eta_l = jb(le); eta_l=eta_l.astype(cp.float32)
    _, phi_l = jb(lp); phi_l=phi_l.astype(cp.float32)
    nev=off_j.size-1; J=pt_j.size
    n_jet=cp.diff(off_j); n_lep=cp.diff(off_l)
    eoj=cp.repeat(cp.arange(nev,dtype=cp.int64), n_jet); lpj=n_lep[eoj]
    pair_off=cp.zeros(J+1,cp.int64); cp.cumsum(lpj,out=pair_off[1:]); P=int(pair_off[-1])
    jop=cp.repeat(cp.arange(J,dtype=cp.int64), lpj)
    lop=off_l[eoj[jop]] + (cp.arange(P,dtype=cp.int64)-pair_off[jop])
    return dict(off_j=off_j,pt_j=pt_j,eta_j=eta_j,phi_j=phi_j,eta_l=eta_l,phi_l=phi_l,
                nev=nev,J=J,n_jet=n_jet,lpj=lpj,pair_off=pair_off,jop=jop,lop=lop,
                msmax=int(n_lep.max()) if nev else 1, jmax=int(n_jet.max()) if nev else 1)

def compute_array(d):
    deta=d['eta_j'][d['jop']]-d['eta_l'][d['lop']]
    dphi=d['phi_j'][d['jop']]-d['phi_l'][d['lop']]
    dphi=(dphi+np.float32(MPI))%(np.float32(2*MPI))-np.float32(MPI)
    dr=cp.sqrt(deta*deta+dphi*dphi).astype(cp.float32)
    drmin=cp.empty(d['J'],cp.float32)
    segmented_reduce(d_in=dr,d_out=drmin,num_segments=d['J'],start_offsets_in=d['pair_off'][:-1],
                     end_offsets_in=d['pair_off'][1:],op=OpKind.MINIMUM,h_init=np.array([np.inf],np.float32),max_segment_size=d['msmax'])
    drmin[d['lpj']==0]=np.float32(-1.0)
    contrib=cp.where((d['pt_j']>30)&(drmin>0.4),d['pt_j'],cp.float32(0)).astype(cp.float32)
    ht=cp.empty(d['nev'],cp.float32)
    segmented_reduce(d_in=contrib,d_out=ht,num_segments=d['nev'],start_offsets_in=d['off_j'][:-1],
                     end_offsets_in=d['off_j'][1:],op=OpKind.PLUS,h_init=np.array([0],np.float32),max_segment_size=d['jmax'])
    return ht

def compute_perm(d):
    ej,el,pj,pl,jop,lop=d['eta_j'],d['eta_l'],d['phi_j'],d['phi_l'],d['jop'],d['lop']
    def dr_op(t) -> np.float32:                       # stateless: reads the zipped tuple only
        de=t[0]-t[1]; dp=t[2]-t[3]
        dp=(dp+MPI)%(2.0*MPI)-MPI
        return sqrt(de*de+dp*dp)
    z=ZipIterator(PermutationIterator(ej,jop),PermutationIterator(el,lop),
                  PermutationIterator(pj,jop),PermutationIterator(pl,lop))
    drmin=cp.empty(d['J'],cp.float32)
    segmented_reduce(d_in=TransformIterator(z,dr_op),d_out=drmin,num_segments=d['J'],
                     start_offsets_in=d['pair_off'][:-1],end_offsets_in=d['pair_off'][1:],
                     op=OpKind.MINIMUM,h_init=np.array([np.inf],np.float32),max_segment_size=d['msmax'])
    drmin[d['lpj']==0]=np.float32(-1.0)
    pt_j=d['pt_j']
    def contrib_op(t) -> np.float32:                  # stateless: pt if (pt>30 and drmin>0.4) else 0
        return t[0] if (t[0]>30.0 and t[1]>0.4) else 0.0
    zc=ZipIterator(pt_j, drmin)
    ht=cp.empty(d['nev'],cp.float32)
    segmented_reduce(d_in=TransformIterator(zc,contrib_op),d_out=ht,num_segments=d['nev'],
                     start_offsets_in=d['off_j'][:-1],end_offsets_in=d['off_j'][1:],
                     op=OpKind.PLUS,h_init=np.array([0],np.float32),max_segment_size=d['jmax'])
    return ht

def t(fn,d,n=7):
    ts=[]
    for _ in range(n):
        cp.cuda.Device(0).synchronize(); t0=time.perf_counter(); fn(d); cp.cuda.Device(0).synchronize()
        ts.append(time.perf_counter()-t0)
    return statistics.median(ts[1:])*1e3

for s in ["1M","10M"]:
    d=load(f"../data/pq_subset_{s}.parquet")
    a=compute_array(d); b=compute_perm(d)
    md=float(cp.abs(a-b).max())
    print(f"{s}: match={'OK' if md<1e-3 else 'DIFF'} (max|diff|={md:.2e})  compute  array {t(compute_array,d):6.2f} ms   perm-fused {t(compute_perm,d):6.2f} ms")
