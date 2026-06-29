"""GPU-direct reader: cudf.read_parquet -> awkward (cuda backend), no host round-trip.

Drop-in alternative to cpu_read_shim. Selected by AK_BENCH_READER=cudf in
run_adl_queries.py. Requires cudf installed in the env (see cudf_inject_overrides.txt
at the repo root for how cudf 26.6 was forced in alongside cuda.compute).

Unlike cpu_read_shim (pyarrow CPU read + ak.Array + to_backend), here the parquet
decode happens on the GPU (libcudf) and the resulting device buffers are wrapped as
awkward content WITHOUT copying to host -- so the read/load timing is GPU-direct and
comparable to the compute stage.

Handles the column shapes the ADL benchmark uses: flat numeric (MET_pt, ...) and
1-level jagged numeric (Jet_pt, Muon_*, Electron_*, ...). It deliberately does NOT
reuse ak_from_cudf.py: that module pokes at cudf internals (base_mask/base_children)
that were renamed in cudf 26.6. This converter sticks to stable public column API
(.elements, .count_elements(), .data) so it survives cudf version drift.
"""
import cupy
import numpy
import cudf
import awkward as ak

__version__ = f"gpu_read_cudf (cudf {cudf.__version__}, GPU-direct)"


def read_parquet(filepath, columns=None):
    """Stand-in for cudf.read_parquet -> returns a cudf DataFrame (GPU-resident)."""
    return cudf.read_parquet(filepath, columns=columns)


def cudf_to_awkward(col):
    """Convert a cudf Series to an awkward Array on the CUDA backend, zero-copy.

    `col` is a cudf Series (from df["name"]), matching how the benchmark indexes.
    """
    column = col._data[col.name]

    if isinstance(column, cudf.core.column.ListColumn):
        # 1-level jagged numeric: rebuild offsets from per-row counts (public API),
        # wrap the leaf device buffer directly.
        if column.null_count != 0 or column.elements.null_count != 0:
            raise NotImplementedError(
                "gpu_read_cudf: option-typed/list-with-nulls not supported; "
                "ADL benchmark columns are non-nullable"
            )
        counts = cupy.asarray(column.count_elements().values)
        offsets = cupy.zeros(counts.size + 1, dtype=cupy.int64)
        cupy.cumsum(counts, out=offsets[1:])
        leaf = column.elements
        n = int(offsets[-1].item())
        data = cupy.asarray(leaf.data).view(numpy.dtype(leaf.dtype))[:n]
        content = ak.contents.NumpyArray(data)
        return ak.Array(
            ak.contents.ListOffsetArray(ak.index.Index64(offsets), content)
        )

    # flat numeric column
    return ak.from_cupy(col.to_cupy())
