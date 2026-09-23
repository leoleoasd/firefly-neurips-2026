#!/bin/bash
# Download and convert HuggingFace models to torch_dist format for slime training
#
# Usage:
#   ./download_convert_model.sh [OPTIONS]
#
# Options:
#   -m, --model       HuggingFace model name (e.g., zai-org/GLM-Z1-9B-0414)
#   -c, --config      Model config name from scripts/models/ (e.g., glm4-9B, qwen3-4B)
#   -o, --output      Output directory for converted model (default: /tmp/instance_storage/<model_name>_torch_dist)
#   -d, --download    Download directory for HF model (default: /tmp/instance_storage/<model_name>)
#   -h, --help        Show this help message
#
# Environment variables (can be used instead of options):
#   HF_MODEL_NAME     HuggingFace model name
#   MODEL_CONFIG      Model config name
#   SAVE_PATH         Output directory for converted model
#   DOWNLOAD_PATH     Download directory for HF model
#
# Examples:
#   ./download_convert_model.sh -m zai-org/GLM-Z1-9B-0414 -c glm4-9B
#   ./download_convert_model.sh --model Qwen/Qwen3-4B --config qwen3-4B
#   HF_MODEL_NAME=zai-org/GLM-Z1-9B-0414 MODEL_CONFIG=glm4-9B ./download_convert_model.sh

set -e

source /workdir/.venv/bin/activate

SLIME_DIR="/workdir/thirdparty/slime"
MODELS_CONFIG_DIR="${SLIME_DIR}/scripts/models"
DEFAULT_STORAGE="/tmp/instance_storage"

# Verify slime directory exists
if [[ ! -d "${SLIME_DIR}" ]]; then
    echo "Error: Slime directory not found at: ${SLIME_DIR}"
    echo "Set SLIME_DIR environment variable to override."
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            HF_MODEL_NAME="$2"
            shift 2
            ;;
        -c|--config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            SAVE_PATH="$2"
            shift 2
            ;;
        -d|--download)
            DOWNLOAD_PATH="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -28
            echo ""
            echo "Available model configs:"
            ls -1 "${MODELS_CONFIG_DIR}" 2>/dev/null | sed 's/\.sh$//' | sed 's/^/  /'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "${HF_MODEL_NAME}" ]]; then
    echo "Error: HuggingFace model name is required."
    echo "Use -m/--model or set HF_MODEL_NAME environment variable."
    echo "Use -h or --help for usage information."
    exit 1
fi

if [[ -z "${MODEL_CONFIG}" ]]; then
    echo "Error: Model config is required."
    echo "Use -c/--config or set MODEL_CONFIG environment variable."
    echo ""
    echo "Available model configs:"
    ls -1 "${MODELS_CONFIG_DIR}" 2>/dev/null | sed 's/\.sh$//' | sed 's/^/  /'
    exit 1
fi

# Derive model name from HF path (last component)
MODEL_NAME=$(basename "${HF_MODEL_NAME}")

# Set default paths if not specified
DOWNLOAD_PATH="${DOWNLOAD_PATH:-${DEFAULT_STORAGE}/${MODEL_CONFIG}}"
SAVE_PATH="${SAVE_PATH:-${DEFAULT_STORAGE}/${MODEL_CONFIG}_torch_dist}"

# Validate model config exists
CONFIG_FILE="${MODELS_CONFIG_DIR}/${MODEL_CONFIG}.sh"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Model config not found: ${CONFIG_FILE}"
    echo ""
    echo "Available model configs:"
    ls -1 "${MODELS_CONFIG_DIR}" 2>/dev/null | sed 's/\.sh$//' | sed 's/^/  /'
    exit 1
fi

echo "============================================"
echo "Download and Convert Model"
echo "============================================"
echo "HuggingFace Model: ${HF_MODEL_NAME}"
echo "Model Config:      ${MODEL_CONFIG}"
echo "Download Path:     ${DOWNLOAD_PATH}"
echo "Output Path:       ${SAVE_PATH}"
echo "Config File:       ${CONFIG_FILE}"
echo "============================================"

# Create storage directory if needed
mkdir -p "${DEFAULT_STORAGE}"

# Download model from HuggingFace
echo ""
echo "[1/2] Downloading model from HuggingFace..."
hf download "${HF_MODEL_NAME}" --local-dir "${DOWNLOAD_PATH}"

# Source the model config to get MODEL_ARGS
echo ""
echo "[2/2] Converting model to torch_dist format..."
cd "${SLIME_DIR}"
source "${CONFIG_FILE}"

python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${DOWNLOAD_PATH}" \
    --save "${SAVE_PATH}"

echo ""
echo "============================================"
echo "Conversion complete!"
echo "Converted model saved to: ${SAVE_PATH}"
echo "============================================"
