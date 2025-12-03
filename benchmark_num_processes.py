"""
Benchmark script to find optimal num_processes for your hardware.
Runs short training sessions with different num_processes values and measures throughput.
"""

import os
import sys
import time
import yaml
import tempfile
import shutil
from pathlib import Path

# Test configurations
TEST_PROCESSES = [12, 16, 20, 24, 32, 40, 48]
TEST_EPOCHS = 5  # Short test
TEST_STEP_PER_EPOCH = 5000  # Reduced for speed

def run_benchmark(config_path, num_processes):
    """Run training with specific num_processes and measure speed"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify config for benchmarking
    config['train']['num_processes'] = num_processes
    config['train']['epoch'] = TEST_EPOCHS
    config['train']['step_per_epoch'] = TEST_STEP_PER_EPOCH
    config['log_interval'] = 999  # Disable logging

    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config = tmp.name

    # Create temporary log directory
    tmp_log = tempfile.mkdtemp(prefix='benchmark_')

    try:
        print(f"\nTesting num_processes={num_processes}...")

        start_time = time.time()

        # Run training
        cmd = f"python ts_train.py --config {tmp_config} 2>&1 | grep -E 'epoch|step_per_sec'"
        result = os.system(cmd)

        elapsed = time.time() - start_time

        if result == 0:
            total_samples = TEST_EPOCHS * TEST_STEP_PER_EPOCH
            samples_per_sec = total_samples / elapsed
            time_per_epoch = elapsed / TEST_EPOCHS

            print(f"  ✓ Completed in {elapsed:.1f}s")
            print(f"  → {samples_per_sec:.0f} samples/sec")
            print(f"  → {time_per_epoch:.1f}s per epoch")

            return {
                'num_processes': num_processes,
                'elapsed': elapsed,
                'samples_per_sec': samples_per_sec,
                'time_per_epoch': time_per_epoch
            }
        else:
            print(f"  ✗ Failed")
            return None

    finally:
        # Cleanup
        os.unlink(tmp_config)
        if os.path.exists(tmp_log):
            shutil.rmtree(tmp_log)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_num_processes.py <config_file>")
        print("Example: python benchmark_num_processes.py cfg/config_multisize_curriculum_by_size.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("="*60)
    print("Benchmarking num_processes for optimal training speed")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Test epochs: {TEST_EPOCHS}")
    print(f"Steps per epoch: {TEST_STEP_PER_EPOCH}")
    print(f"Testing values: {TEST_PROCESSES}")
    print("="*60)

    results = []

    for num_proc in TEST_PROCESSES:
        result = run_benchmark(config_path, num_proc)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause between tests

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Processes':<12} {'Samples/sec':<15} {'Sec/epoch':<12} {'Total time':<12}")
    print("-"*60)

    for r in results:
        print(f"{r['num_processes']:<12} {r['samples_per_sec']:<15.0f} "
              f"{r['time_per_epoch']:<12.1f} {r['elapsed']:<12.1f}")

    # Find best
    if results:
        best = max(results, key=lambda x: x['samples_per_sec'])
        print("="*60)
        print(f"FASTEST: num_processes={best['num_processes']} "
              f"({best['samples_per_sec']:.0f} samples/sec)")
        print("="*60)
        print(f"\nRecommendation: Set 'num_processes: {best['num_processes']}' in your config")
