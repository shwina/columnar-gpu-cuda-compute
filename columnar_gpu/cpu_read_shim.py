"""Drop-in replacement for the `cudf` read path used by run_adl_queries.py.

WHY: cudf 25.12 hard-pins numba-cuda 0.19.x (old cuda-core 0.3.x API), which is
incompatible with cuda.compute (needs cuda-core 1.0.x). So cudf and the awkward3
cuda.compute backend cannot coexist in one environment. To compare the GPU
*compute* backend (CuPy RawKernel vs cuda.compute) on an identical read path, we
drop cudf entirely and read parquet with pyarrow, then move to the CUDA backend.

This mirrors the benchmark's own CPU path (`ak.Array(pq_table["col"])`) so the
GPU and CPU branches use the same arrow->awkward conversion semantics; the only
added step is `ak.to_backend(..., "cuda")`. NOTE: the read/load *timings* are not
comparable to cudf's GPU-direct read — only the compute stage is the point here.
"""

import pyarrow.parquet as _pq
import awkward as ak

__version__ = "cpu_read_shim-1.0 (pyarrow, no cudf)"


def read_parquet(filepath, columns=None):
    """Stand-in for cudf.read_parquet -> returns a pyarrow Table.

    A pyarrow Table supports table["col"] (returns a ChunkedArray), matching how
    the benchmark indexes the cudf DataFrame.
    """
    return _pq.read_table(filepath, columns=columns)


def cudf_to_awkward(col):
    """Stand-in for ak_from_cudf.cudf_to_awkward.

    `col` is a pyarrow (Chunked)Array obtained from table["name"]. Convert exactly
    as the CPU path does (`ak.Array(col)`) and move it to the CUDA backend.
    """
    return ak.to_backend(ak.Array(col), "cuda")
