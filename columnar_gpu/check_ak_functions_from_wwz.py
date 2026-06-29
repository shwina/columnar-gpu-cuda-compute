import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector

def run_functions(in_vals_for_arr,backend_name):
    arr = ak.Array(in_vals_for_arr,backend=backend_name)
    print("\nBackend:",backend_name)
    print("01:",ak.fill_none(arr,999))
    print("02:",ak.ones_like(arr))
    print("03:",ak.zeros_like(arr))
    print("04:",ak.full_like(arr,999))
    print("05:",ak.pad_none(arr,10))
    print("06:",ak.mask(arr,arr>2))
    print("07:",ak.concatenate([arr,arr],axis=1))
    print("08:",ak.with_name(ak.concatenate([arr,arr],axis=1),'PtEtaPhiMCandidate'))
    print("09:",ak.argsort(arr,axis=-1))
    print("10:",ak.combinations(arr,3))
    print("11:",ak.argcombinations(arr,3))
    print("12:",ak.cartesian({"a":arr,"b":arr}))
    print("13:",ak.min(arr))
    print("14:",ak.max(arr))
    print("15:",ak.num(arr))
    print("16:",ak.sum(arr))
    print("17:",ak.any(arr))
    print("18:",ak.argmax(arr))
    print("19:",ak.flatten(arr))
    print("20:",ak.unflatten(ak.flatten(arr),ak.num(arr)))
    print("21:",ak.where(arr>2,arr,-999))
    print("22:",ak.local_index(arr))
    print("23:",ak.zip({"pt":arr,"eta":0,"phi":arr,"mass":arr},with_name="PtEtaPhiMLorentzVector",behavior=vector.behavior))
    print("24:",ak.to_numpy(ak.flatten(arr)))
    print("25:",ak.values_astype(arr,np.float32))
    print("26:",ak.broadcast_arrays(arr,arr))


val_lst = [[-1.1,0,1.1,2.1,3.1,4.1,None],[-1.2,0,1.2,2.2,3.2,4.2]]

run_functions(val_lst,"cpu")
run_functions(val_lst,"cuda")
