# Performance and Robustness Benchmark

This directory is the self-contained submission workspace for the manuscript
"Performance and Robustness Benchmarking Across Battery SOC Estimator Classes
for Embedded Applications".

## Contents

- `main.tex`: non-anonymized manuscript with author and funding information
- `main_anonymized.tex`: double-blind build using the same scientific source
- `figures/`: the 26 figures referenced by the manuscript
- `tables/`: the two external LaTeX tables referenced by the manuscript
- `bibliography/references.bib`: local bibliography database
- `submission/`: current title page, highlights, cover letter, and checklist

Raw measurement data, benchmark result archives, model binaries, and plotting
scripts are intentionally not duplicated here. Their public locations are
documented in the manuscript's Data Availability section.

## Build

Run the following commands from this directory:

```powershell
latexmk -pdf main.tex
latexmk -pdf main_anonymized.tex
```

The submission documents can be built from `submission/`:

```powershell
latexmk -pdf Title_Page.tex
latexmk -pdf Highlights.tex
latexmk -pdf Cover_Letter.tex
```

The project uses the Elsevier `elsarticle` class and standard TeX Live
packages. No file outside this directory is required for compilation.

## Submission Use

Use `main_anonymized.pdf` when the editorial system requires a blinded
manuscript. Use `main.pdf` as the complete author version. Before uploading,
review the manual items listed in `submission/CHECKLIST.md`.
