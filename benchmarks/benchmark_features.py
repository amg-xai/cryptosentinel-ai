"""
Feature engineering benchmark.
Compares Python loop vs NumPy vs Numba for velocity computation.
Run: python benchmarks/benchmark_features.py
"""
import time
import numpy as np
import sys
sys.path.insert(0, ".")

from src.ml.tabular.vectorized_features import (
    compute_velocity_numba,
    compute_velocity_numpy,
    warmup_jit,
)


def compute_velocity_python(timestamps, window_seconds):
    n = len(timestamps)
    counts = []
    for i in range(n):
        count = sum(
            1 for j in range(n)
            if 0 <= timestamps[i] - timestamps[j] <= window_seconds
        )
        counts.append(count)
    return counts


def benchmark(name, fn, *args, n_runs=5):
    fn(*args)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    mean_ms = np.mean(times) * 1000
    print(f"  {name:30s}: {mean_ms:8.2f}ms")
    return mean_ms


def main():
    print("Warming up JIT compilation...")
    warmup_jit()
    print("JIT ready.\n")

    for n_samples in [100, 1000, 5000]:
        print(f"=== N={n_samples} transactions ===")
        timestamps_list = sorted(np.random.uniform(0, 7200, n_samples).tolist())
        timestamps_np = np.array(timestamps_list, dtype=np.float64)
        window = 3600.0

        if n_samples <= 1000:
            t_python = benchmark("Python loop", compute_velocity_python,
                                 timestamps_list, window)
        else:
            t_python = None
            print(f"  {'Python loop':30s}: skipped (too slow)")

        t_numpy = benchmark("NumPy vectorized", compute_velocity_numpy,
                            timestamps_np, window)
        t_numba = benchmark("Numba JIT (parallel)", compute_velocity_numba,
                            timestamps_np, window)

        if t_python:
            print(f"  NumPy speedup:  {t_python/t_numpy:.1f}x")
            print(f"  Numba speedup:  {t_python/t_numba:.1f}x")
        print()


if __name__ == "__main__":
    main()
