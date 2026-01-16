#!/bin/bash
# Phase 1 Experiment Runner
# Runs experiments via Docker with proper volume mounts

set -e

TOOL_DIR="/home/h.baker@hertie-school.lan/workspace/thesis_analysis/tools/llm-efficiency-measurement-tool"
EXPERIMENTS_DIR="/home/h.baker@hertie-school.lan/workspace/thesis_analysis/experiments/phase1"

# Usage check
if [ -z "$1" ]; then
    echo "Usage: $0 <config.yaml> [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/1a_sanity/baseline.yaml"
    echo "  $0 configs/1c_parallelism/vllm_tp_2gpu.yaml --backend vllm"
    echo ""
    echo "Available configs:"
    find "${EXPERIMENTS_DIR}/configs" -name "*.yaml" | sed "s|${EXPERIMENTS_DIR}/||"
    exit 1
fi

CONFIG_PATH="$1"
shift  # Remove first arg, remaining args passed to llm-energy-measure

# Check if vLLM backend is needed
if [[ "$CONFIG_PATH" == *"vllm"* ]] || [[ "$*" == *"--backend vllm"* ]]; then
    SERVICE="vllm"
    echo "Using vLLM backend..."
else
    SERVICE="pytorch"
    echo "Using PyTorch backend..."
fi

# Run the experiment
cd "${TOOL_DIR}"
docker compose run --rm \
    -v "${EXPERIMENTS_DIR}/configs:/app/experiment_configs:ro" \
    -v "${EXPERIMENTS_DIR}/results:/app/results" \
    ${SERVICE} \
    llm-energy-measure experiment "/app/experiment_configs/${CONFIG_PATH#configs/}" "$@"
