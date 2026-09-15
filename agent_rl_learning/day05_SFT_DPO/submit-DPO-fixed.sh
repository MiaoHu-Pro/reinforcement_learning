#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=qwen3-0.6b-dpo
# Slurm does not expand ~ or $HOME inside #SBATCH directives. Submit this
# script from day05_SFT_DPO so the relative output path resolves correctly.
#SBATCH --output=result_out/qwen3-0.6b-dpo-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/rl_learning_demo/day05_SFT_DPO"
SFT_MODEL_PATH="${HOME}/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT"
DPO_OUTPUT_PATH="${HOME}/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT-DPO"
CONDA_ENV_NAME="rl_post_training_env"

# `conda activate` is normally unavailable in a non-interactive Slurm shell
# until Conda's shell integration has been sourced.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"

# The model and preference dataset are local. A job-specific cache avoids lock
# contention if multiple Slurm jobs use Hugging Face datasets simultaneously.
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen3-dpo-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${HF_HOME}"
cd "${TRAINING_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Input SFT model: ${SFT_MODEL_PATH}"
echo "DPO output model: ${DPO_OUTPUT_PATH}"
"${PYTHON_EXECUTABLE}" --version

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi

if [[ ! -f "${TRAINING_DIR}/2-DPO-fixed.py" ]]; then
    echo "Training script not found: ${TRAINING_DIR}/2-DPO-fixed.py" >&2
    exit 1
fi

if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
    echo "SFT model directory not found: ${SFT_MODEL_PATH}" >&2
    exit 1
fi

# Show the allocated GPU and fail immediately if this PyTorch installation
# cannot use it. A CUDA-enabled PyTorch wheel is required for DPO training.
nvidia-smi
"${PYTHON_EXECUTABLE}" -c \
    "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'"

# 2-DPO-fixed.py reads all local train_prefs records, initializes both the
# policy and frozen reference from Qwen3-0.6B-SFT, and saves only the trained
# policy/tokenizer to Qwen3-0.6B-SFT-DPO.
srun "${PYTHON_EXECUTABLE}" 2-DPO-fixed.py

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved model: ${DPO_OUTPUT_PATH}"
