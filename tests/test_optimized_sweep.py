from rocmGPUBenches import GPUCacheBenchmark
import time

print("Creating GPU Cache Benchmark...")
start = time.time()
bench = GPUCacheBenchmark()
print(f"Benchmark created in {time.time() - start:.2f}s (includes one-time kernel compilation)")
print()

print(f"GPU: {bench.get_device_name()}")
print(f"Compute Units: {bench.get_sm_count()}")
print()

print("Running sweep test (should NOT recompile for each N value)...")
sweep_start = time.time()
results = bench.run_sweep()
sweep_time = time.time() - sweep_start

print()
print(f"Sweep completed in {sweep_time:.2f}s")
print()
print("Results:")
print(f"{'N':<8} {'Data (KB)':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Spread %':<10}")
print("-" * 70)
for r in results:
    print(f"{r.N:<8} {r.data_size_kb:<12.1f} {r.exec_time_ms:<12.3f} {r.bandwidth_gbs:<18.0f} {r.spread_percent:<10.2f}")
