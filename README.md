# LLM Efficiency Measurement: Thesis Analysis

This repository contains the complete thesis, analysis scripts, and experimental data for my MDS thesis on LLM energy efficiency measurement.

## Repository Structure

```
├── thesis/                     # Thesis document
│   ├── henry_baker_thesis.pdf  # Compiled thesis PDF
│   └── src/                    # LaTeX source files
├── analysis/                   # Analysis code
│   ├── notebooks/              # Jupyter notebooks for visualisation
│   └── scripts/                # Python analysis scripts
├── data/                       # Experimental results
│   ├── controlled_results.csv  # Controlled experiment data
│   ├── grid_results.csv        # Grid search experiment data
│   └── scenarios_results.csv   # Scenario analysis data
└── tools/                      # Submodules
    └── llm-efficiency-measurement-tool/  # The measurement tool (v1.0.0)
```

## Related Repository

The experimental data was generated using the [LLM Efficiency Measurement Tool](https://github.com/henrycgbaker/llm-efficiency-measurement-tool) (included as a submodule at v1.0.0).

## Cloning

To clone with the submodule:

```bash
git clone --recurse-submodules git@github.com:henrycgbaker/thesis_analysis.git
```

Or if already cloned:

```bash
git submodule update --init --recursive
```
