# Results Workspace

This directory is reserved for the paper-facing result package.

Structure:

- `tables/`
  - compact comparison tables
  - scenario-wise ranking tables
  - local recovery tables
- `figures/`
  - overview plots
  - scenario-specific zoom plots
  - recovery figures
- `build_paper_results.py`
  - assembles the result package from the completed benchmark campaigns

Current intended figure set:

1. baseline comparison across models
2. delta-MAE heatmap across scenarios
3. current-offset focus figure
4. missing-samples / missing-gap focus figure
5. local initial-state recovery figure
6. local current-noise drift figure
7. spike robustness figure

Current intended table set:

1. baseline table
2. scenario delta table
3. local recovery table
4. model ranking by scenario
