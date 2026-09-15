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
#SBATCH --job-name=gpt2-chinese-rm
# Slurm does not expand ~ or $HOME in #SBATCH directives. Submit this script
# from day06_InstructGPT so this relative output path resolves correctly.
#SBATCH --output=result_out/gpt2-chinese-rm-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/rl_learning_demo/day06_InstructGPT"
SFT_MODEL_PATH="${HOME}/scratch/llms_model/post_trained_models/gpt2-chinese-cluecorpussmall-sft"
RM_OUTPUT_PATH="${HOME}/scratch/llms_model/post_trained_models/gpt2-chinese-cluecorpussmall-reward-model"
DATA_PATH="${PROJECT_ROOT}/data/online_shopping_10_cats.csv"
CONDA_ENV_NAME="rl_post_training_env"

# Make `conda activate` available inside the non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/gpt2-chinese-rm-${SLURM_JOB_ID}"
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
echo "Reward-model output: ${RM_OUTPUT_PATH}"
"${PYTHON_EXECUTABLE}" --version

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi

if [[ ! -f "${TRAINING_DIR}/2-RM-fixed.py" ]]; then
    echo "Training script not found: ${TRAINING_DIR}/2-RM-fixed.py" >&2
    exit 1
fi

# Reward-model training must start from the completed SFT checkpoint.
if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
    echo "SFT model directory not found: ${SFT_MODEL_PATH}" >&2
    echo "Run submit-SFT-fixed.sh successfully before this job." >&2
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "Training CSV not found: ${DATA_PATH}" >&2
    exit 1
fi

nvidia-smi
"${PYTHON_EXECUTABLE}" -c \
    "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'"

srun "${PYTHON_EXECUTABLE}" 2-RM-fixed.py

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved reward model: ${RM_OUTPUT_PATH}/reward_model.pt"
