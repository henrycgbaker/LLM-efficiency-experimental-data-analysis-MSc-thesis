#!/bin/bash
# Phase 1 Batch Runner
# Runs all experiments in a phase subdirectory

TOOL_DIR="/home/h.baker@hertie-school.lan/workspace/thesis_analysis/tools/llm-efficiency-measurement-tool"
EXPERIMENTS_DIR="/home/h.baker@hertie-school.lan/workspace/thesis_analysis/experiments/phase1"
REPO_ROOT="/home/h.baker@hertie-school.lan/workspace/thesis_analysis"

# Source .env from repo root (for HF_TOKEN, etc.)
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a  # auto-export
    source "${REPO_ROOT}/.env"
    set +a
fi

# TODO: Enable user mapping once tool fixes state dir permissions
# export DOCKER_UID=$(id -u)
# export DOCKER_GID=$(id -g)

# Usage check
if [ -z "$1" ]; then
    echo "Usage: $0 <phase> [--dry-run] [--skip-errors]"
    echo ""
    echo "Phases:"
    echo "  1a_sanity      - Parameter sanity checks (~30 mins)"
    echo "  1b_features    - New feature validation (~2 hours)"
    echo "  1c_parallelism - Parallelism deep-dive (~4-5 hours)"
    echo ""
    echo "Options:"
    echo "  --dry-run      Show config without running"
    echo "  --skip-errors  Continue on experiment failures"
    echo ""
    echo "Examples:"
    echo "  $0 1a_sanity"
    echo "  $0 1b_features --dry-run"
    echo "  $0 1a_sanity --skip-errors"
    exit 1
fi

PHASE="$1"
DRY_RUN=""
SKIP_ERRORS=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --skip-errors) SKIP_ERRORS=true ;;
    esac
done

CONFIG_DIR="${EXPERIMENTS_DIR}/configs/${PHASE}"
RESULTS_BASE="${EXPERIMENTS_DIR}/results/${PHASE}"

if [ ! -d "${CONFIG_DIR}" ]; then
    echo "Error: Phase directory not found: ${CONFIG_DIR}"
    exit 1
fi

# Create results directory
mkdir -p "${RESULTS_BASE}"

echo "========================================"
echo "Running Phase: ${PHASE}"
echo "Config dir: ${CONFIG_DIR}"
echo "Results: ${RESULTS_BASE}"
echo "========================================"

# Count configs
CONFIG_COUNT=$(find "${CONFIG_DIR}" -name "*.yaml" | wc -l)
echo "Found ${CONFIG_COUNT} configurations"
echo ""

# Track results
SUCCEEDED=0
FAILED=0
FAILED_CONFIGS=""

# Run each config
COUNTER=0
for config in "${CONFIG_DIR}"/*.yaml; do
    COUNTER=$((COUNTER + 1))
    CONFIG_NAME=$(basename "${config}" .yaml)
    CONFIG_RESULTS="${RESULTS_BASE}/${CONFIG_NAME}"

    # Skip if results already exist (check for multi_cycle result with flexible naming)
    if [ -d "${CONFIG_RESULTS}/multi_cycle" ] && ls "${CONFIG_RESULTS}/multi_cycle/"*.json >/dev/null 2>&1; then
        echo "----------------------------------------"
        echo "[${COUNTER}/${CONFIG_COUNT}] Skipping: ${CONFIG_NAME} (results exist)"
        echo "----------------------------------------"
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi

    echo "----------------------------------------"
    echo "[${COUNTER}/${CONFIG_COUNT}] Running: ${CONFIG_NAME}"
    echo "----------------------------------------"

    # Determine backend service
    if [[ "${config}" == *"vllm"* ]]; then
        SERVICE="vllm"
    else
        SERVICE="pytorch"
    fi

    # Use docker compose with configurable paths via environment variables
    # Compose handles: user mapping, privileged mode, env vars, volumes
    if LLM_ENERGY_CONFIGS_DIR="${CONFIG_DIR}" \
       LLM_ENERGY_RESULTS_DIR="${RESULTS_BASE}" \
       docker compose -f "${TOOL_DIR}/docker-compose.yml" run --rm \
        ${SERVICE} \
        llm-energy-measure experiment "/app/configs/${CONFIG_NAME}.yaml" \
            --results-dir "/app/results/${CONFIG_NAME}" \
            --yes ${DRY_RUN}; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "✓ ${CONFIG_NAME} complete"
    else
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS="${FAILED_CONFIGS} ${CONFIG_NAME}"
        echo "✗ ${CONFIG_NAME} failed"
        if [ "$SKIP_ERRORS" = false ]; then
            echo "Stopping due to error. Use --skip-errors to continue."
            break
        fi
    fi

    echo ""
done

echo "========================================"
echo "Phase ${PHASE} complete!"
echo "Succeeded: ${SUCCEEDED}/${CONFIG_COUNT}"
if [ ${FAILED} -gt 0 ]; then
    echo "Failed: ${FAILED} -${FAILED_CONFIGS}"
fi
echo "Results saved to: ${RESULTS_BASE}"
echo "========================================"
