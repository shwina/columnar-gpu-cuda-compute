import time
#import awkward as ak
#import cupy as cp
#import numpy as np
#import numba as nb
#
#import torch
#import cudf
#from ak_from_cudf import cudf_to_awkward
#
#import pyarrow.parquet as pq
#import fastparquet
#from hepconvert import root_to_parquet
#
#import uproot
#from coffea.jitters import hist as gpu_hist
#import hist
#from coffea.nanoevents.methods import candidate

import cudf
import pandas as df
import numpy as np

filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"

#######################################
print("\n--- cudf DF ---\n")

t0 = time.time()
cudf_x = cudf.read_parquet(filepath, columns = ["MET_pt"])["MET_pt"]
t1 = time.time()

t2 = time.time()
cudf_x = cudf_x + 1
t3 = time.time()

t4 = time.time()
cudf_x = cudf_x + cudf.Series(list(np.arange(len(cudf_x))))
t5 = time.time()

print(type(cudf_x),len(cudf_x))
print(len(cudf_x))
print(t1-t0)
print(t3-t2)
print(t5-t4)

#######################################
print("\n--- Regular DF ---\n")

t0 = time.time()
df_x = df.read_parquet(filepath, columns = ["MET_pt"])["MET_pt"]
t1 = time.time()

t2 = time.time()
df_x = df_x + 1
t3 = time.time()

t4 = time.time()
df_x = df_x + df.Series(list(np.arange(len(df_x))))
t5 = time.time()

print(type(df_x),len(df_x))
print(len(df_x))
print(t1-t0)
print(t3-t2)
print(t5-t4)

