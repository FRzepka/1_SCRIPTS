# Simulation Environment

Scenario-driven test harness for SOC+SOH models under BMS constraints.

Start with:
- `ROBUSTNESS_BENCHMARK.md`
- `TEST_LOGBOOK.md`
- `SOC_SOH_1.6.0.0_0.1.2.3/TEST_PLAN.md`
- `SOC_SOH_1.6.0.0_0.1.2.3/run_soc_soh_scenario.py`
- `CC_1.0.0/run_cc_scenario.py` (uses CC model block)
- `CC_SOH_1.0.0/run_cc_soh_scenario.py` (uses CC + SOH GRU 0.3.1.2)
- `ECM_0.0.1/run_ecm_scenario.py` (uses ECM_Model native library)
- `robustness_common.py` (shared measurement disturbance logic + metrics)
- `run_minimal_robustness_matrix.py` (compact campaign runner)
- `run_extended_robustness_matrix.py` (extended campaign runner)
- `build_robustness_benchmark_overview.py` (latest-run overview table)
