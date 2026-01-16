# Analysis of LLM Efficiency Measurement at Inference Time

[![Read the Thesis](https://img.shields.io/badge/📄_Read_Thesis-PDF-blue?style=for-the-badge)](thesis/henry_baker_thesis.pdf)
[![Submitted](https://img.shields.io/badge/Submitted-May_2025-green?style=for-the-badge)](thesis/henry_baker_thesis.pdf)

> **Thesis title: The Implementation Gap**: Inducing Variation in LLM Inference-time Energy Efficiency for Fixed Computational Workloads

This repository contains the complete thesis, analysis scripts, and experimental data for my MSc thesis measuring the effect of selected implementation parameters on the energy efficiencies of open-source LLMs at inference time.

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
