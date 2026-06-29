import os
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pandas as df

def main():

    nevents = 10e6
    name = "10M"

    # File paths
    filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
    outpath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets"

    # Dump just the first nevents from Lindsey's file into a smaller file
    pf = ParquetFile(filepath)
    first_ten_rows = next(pf.iter_batches(batch_size = nevents))
    rows_df = pa.Table.from_batches([first_ten_rows]).to_pandas()
    rows_df.to_parquet(os.path.join(outpath,f"pq_subset_{name}.parquet"))
    print("Done")

    # Print the number of events in the files
    #filepath_to_check = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_100k.parquet"
    #filepath_to_check = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_1M.parquet"
    #filepath_to_check = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_10M.parquet"
    filepath_to_check = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
    n_in_filepath_to_check = len(df.read_parquet(filepath_to_check, columns = ["MET_pt"]))
    print(f"\nNumber of nevents: {n_in_filepath_to_check}")

main()
