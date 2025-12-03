"""
Simple benchmark to find optimal num_processes.
Measures raw collection speed without full training overhead.
"""

import sys
import time
import gymnasium as gym
import tianshou as ts
from tools import registration_envs, set_seed
import arguments

# Test different process counts
TEST_PROCESSES = [12, 16, 20, 24, 32, 40, 48]
COLLECT_STEPS = 10000  # Steps to collect for each test


def benchmark_num_processes(args, test_values):
    """Benchmark different num_processes values"""

    results = []

    for num_proc in test_values:
        print(f"\n{'='*60}")
        print(f"Testing num_processes = {num_proc}")
        print(f"{'='*60}")

        try:
            # Create environments
            from envs.Packing.multiSizeWrapper import MultiSizeWrapper

            use_multi_size = args.env.get('bin_sizes') is not None
            bin_sizes = args.env.bin_sizes if use_multi_size else None
            base_container_size = bin_sizes[0] if use_multi_size else args.env.container_size

            def make_env():
                env = gym.make(args.env.id,
                              container_size=base_container_size,
                              enable_rotation=args.env.rot,
                              data_type=args.env.box_type,
                              item_set=args.env.box_size_set,
                              reward_type=args.train.reward_type,
                              action_scheme=args.env.scheme,
                              k_placement=args.env.k_placement)
                if use_multi_size:
                    env = MultiSizeWrapper(env, bin_sizes)
                return env

            # Create vectorized env
            train_envs = ts.env.SubprocVectorEnv(
                [make_env for _ in range(num_proc)]
            )

            # Create dummy random policy for testing
            from tianshou.policy import RandomPolicy
            policy = RandomPolicy()

            # Create collector
            from mycollector import PackCollector
            collector = PackCollector(policy, train_envs)

            # Warmup
            print("Warming up...")
            collector.collect(n_step=100)
            collector.reset()

            # Benchmark
            print(f"Collecting {COLLECT_STEPS} steps...")
            start_time = time.time()
            result = collector.collect(n_step=COLLECT_STEPS)
            elapsed = time.time() - start_time

            # Calculate metrics
            steps_per_sec = COLLECT_STEPS / elapsed
            time_per_1k_steps = elapsed / (COLLECT_STEPS / 1000)

            print(f"  ✓ Completed in {elapsed:.2f}s")
            print(f"  → {steps_per_sec:.0f} steps/sec")
            print(f"  → {time_per_1k_steps:.2f}s per 1000 steps")

            results.append({
                'num_processes': num_proc,
                'steps_per_sec': steps_per_sec,
                'elapsed': elapsed,
                'time_per_1k': time_per_1k_steps
            })

            # Cleanup
            train_envs.close()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

        # Brief pause
        time.sleep(1)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_parallelism.py --config <config_file>")
        print("Example: python benchmark_parallelism.py --config cfg/config_multisize_curriculum_by_size.yaml")
        sys.exit(1)

    # Load config
    registration_envs()
    args = arguments.get_args()
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    print("="*70)
    print(" BENCHMARK: Finding Optimal num_processes")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Collection steps per test: {COLLECT_STEPS}")
    print(f"Testing values: {TEST_PROCESSES}")
    print("="*70)

    # Run benchmark
    results = benchmark_num_processes(args, TEST_PROCESSES)

    # Print summary
    print("\n" + "="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    print(f"{'Processes':<12} {'Steps/sec':<15} {'Sec/1K steps':<15} {'Total time':<12}")
    print("-"*70)

    for r in results:
        print(f"{r['num_processes']:<12} {r['steps_per_sec']:<15.0f} "
              f"{r['time_per_1k']:<15.2f} {r['elapsed']:<12.2f}s")

    # Find best
    if results:
        best = max(results, key=lambda x: x['steps_per_sec'])
        print("="*70)
        print(f" ⭐ FASTEST: num_processes = {best['num_processes']}")
        print(f"    ({best['steps_per_sec']:.0f} steps/sec)")
        print("="*70)
        print(f"\n💡 Recommendation: Update your config with:")
        print(f"   num_processes: {best['num_processes']}")
        print("="*70)
