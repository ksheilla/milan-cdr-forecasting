import glob
import os
import threading
import time

import pandas as pd
import psutil

from config import DATA_RAW_DIR, PARQUET_WRITE_PATH

DATA_DIR = DATA_RAW_DIR
OUTPUT_PARQUET = PARQUET_WRITE_PATH


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class PeakMemoryMonitor:
    """Samples RSS memory in a background thread and tracks the true
    maximum seen during a block of code, instead of relying on
    checkpoint prints that can miss short-lived spikes (e.g. during
    pd.concat)."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop_flag = False
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while not self._stop_flag:
            mem = process.memory_info().rss / (1024 * 1024)
            if mem > self.peak_mb:
                self.peak_mb = mem
            time.sleep(self.interval)

    def start(self):
        self.peak_mb = get_memory_usage_mb()
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        return self.peak_mb


def measure_naive_baseline(sample_file):
    """Loads ONE raw file the naive way (no usecols, no dtype
    downcasting) to get a genuine 'before optimisation' memory number.
    Only run on a single file, since doing this across all 62 files
    is exactly the problem we're avoiding."""
    print("\n--- Baseline (naive) load of one file, for comparison ---")
    mem_before = get_memory_usage_mb()

    naive_df = pd.read_csv(sample_file, sep='\t', header=None)

    mem_after = get_memory_usage_mb()
    naive_mem_used = mem_after - mem_before
    print(f"Naive load memory usage for 1 file: {naive_mem_used:.2f} MB")
    print(f"Naive dtypes:\n{naive_df.dtypes}")

    del naive_df
    return naive_mem_used


def measure_optimized_single_file(sample_file):
    """Loads the SAME single file with the optimized read, for a fair,
    apples-to-apples comparison against measure_naive_baseline()."""
    print("\n--- Optimized load of the same file, for comparison ---")
    mem_before = get_memory_usage_mb()

    df_chunk = pd.read_csv(
        sample_file,
        sep='\t',
        header=None,
        usecols=[0, 1, 7],
        names=['square_id', 'timestamp', 'internet_traffic'],
        dtype={'square_id': 'uint16', 'internet_traffic': 'float32'}
    )
    df_chunk.dropna(subset=['internet_traffic'], inplace=True)

    mem_after = get_memory_usage_mb()
    optimized_mem_used = mem_after - mem_before
    print(f"Optimized load memory usage for 1 file: {optimized_mem_used:.2f} MB")

    del df_chunk
    return optimized_mem_used


def process_and_optimize():
    file_list = sorted(glob.glob(os.path.join(str(DATA_DIR), "sms-call-internet-mi-*")))

    if not file_list:
        print(f"No raw text files found in: {DATA_DIR}")
        print("Download the 62 sms-call-internet-mi-*.txt files from Harvard "
              "Dataverse and place them in that folder. See README.md.")
        return

    # Step A: baseline vs optimized comparison on a single file
    naive_mb = measure_naive_baseline(file_list[0])
    optimized_mb = measure_optimized_single_file(file_list[0])
    reduction_pct = (1 - optimized_mb / naive_mb) * 100 if naive_mb > 0 else 0
    print(f"\nSingle-file memory reduction from optimization: {reduction_pct:.1f}%")

    # Step B: full pipeline, with continuous peak tracking
    print(f"\nInitial Baseline Memory (full run): {get_memory_usage_mb():.2f} MB")

    monitor = PeakMemoryMonitor()
    monitor.start()

    dfs = []
    start_time = time.time()

    for idx, f in enumerate(file_list, 1):
        df_chunk = pd.read_csv(
            f,
            sep='\t',
            header=None,
            usecols=[0, 1, 7],
            names=['square_id', 'timestamp', 'internet_traffic'],
            dtype={
                'square_id': 'uint16',
                'internet_traffic': 'float32'
            }
        )

        df_chunk.dropna(subset=['internet_traffic'], inplace=True)
        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
        df_grouped = df_chunk.groupby(['square_id', 'timestamp'], as_index=False)['internet_traffic'].sum()
        df_grouped['internet_traffic'] = df_grouped['internet_traffic'].astype('float32')

        dfs.append(df_grouped)
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(file_list)} files... Memory: {get_memory_usage_mb():.2f} MB")

    print("\nConcatenating optimized data...")
    full_df = pd.concat(dfs, ignore_index=True)

    full_df.sort_values(by=['square_id', 'timestamp'], inplace=True)

    full_df.to_parquet(OUTPUT_PARQUET, engine='pyarrow', compression='snappy')

    end_time = time.time()
    true_peak_mb = monitor.stop()

    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("DATA PROCESSING & MEMORY OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    print(f"Output Parquet: {OUTPUT_PARQUET}")
    print(f"Optimized Parquet Disk Size: {file_size_mb:.2f} MB")
    print(f"True Peak Memory Usage (continuously sampled): {true_peak_mb:.2f} MB")
    print(f"Total Rows Processed: {len(full_df):,}")
    print(f"Single-file naive vs optimized reduction: {reduction_pct:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    process_and_optimize()
