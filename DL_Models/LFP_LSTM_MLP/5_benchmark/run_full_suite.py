import os
import subprocess
import sys
import time

# --- CONFIGURATION ---
# Paths to the automation scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STM32_BUILD_SCRIPT = os.path.join(SCRIPT_DIR, "STM32", "automate_stm32.py")
SOC_BENCHMARK_SCRIPT = os.path.join(SCRIPT_DIR, "STM32", "SOC", "run_single_benchmark.py")
SOH_BENCHMARK_SCRIPT = os.path.join(SCRIPT_DIR, "STM32", "SOH", "run_single_benchmark.py")
REPORT_SCRIPT = os.path.join(SCRIPT_DIR, "PC", "SOC_SOH_Combined_Results", "generate_combined_report.py")

# Number of samples per on-device benchmark run
# Final setting for paper experiments
BENCHMARK_SAMPLES = 10000

# Define the suite
# Format: (Type, ModelName, BenchmarkScript)
SUITE = [
    ("SOC", "Base", SOC_BENCHMARK_SCRIPT),
    ("SOC", "Pruned", SOC_BENCHMARK_SCRIPT),
    ("SOC", "Quantized", SOC_BENCHMARK_SCRIPT),
    ("SOH", "Base", SOH_BENCHMARK_SCRIPT),
    ("SOH", "Pruned", SOH_BENCHMARK_SCRIPT),
    ("SOH", "Quantized", SOH_BENCHMARK_SCRIPT),
]

def run_command(cmd, description):
    print(f"\n>>> {description}...")
    try:
        subprocess.run(cmd, check=True)
        print(f">>> {description} DONE.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR during {description}. Exit code: {e.returncode}")
        return False

def main():
    print("========================================================")
    print("   STARTING FULL BENCHMARK SUITE (Build -> Flash -> Run)")
    print("========================================================")

    for type_, model, bench_script in SUITE:
        print(f"\n\n--------------------------------------------------------")
        print(f"   PROCESSING: {type_} - {model}")
        print(f"--------------------------------------------------------")

        # Construct the key for automate_stm32.py (e.g., "SOC_Base")
        build_key = f"{type_}_{model}"
        
        if not run_command([sys.executable, STM32_BUILD_SCRIPT, build_key, "all"], f"Build & Flash {build_key}"):
            print("Aborting suite due to flash failure.")
            sys.exit(1)

        # 2. Wait a bit for the board to settle
        time.sleep(2)

        # 3. Run Benchmark (on-device latency / energy / RAM)
        if not run_command(
            [sys.executable, bench_script, model, str(BENCHMARK_SAMPLES)],
            f"Benchmark {model} ({BENCHMARK_SAMPLES} samples)"
        ):
            print("Aborting suite due to benchmark failure.")
            sys.exit(1)

    print("\n\n========================================================")
    print("   GENERATING FINAL REPORT")
    print("========================================================")
    
    if run_command([sys.executable, REPORT_SCRIPT], "Generate Report"):
        print("\n✅ FULL SUITE COMPLETED SUCCESSFULLY!")
    else:
        print("\n⚠️  Suite finished but Report Generation failed.")

if __name__ == "__main__":
    main()
