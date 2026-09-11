#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The private swarm_a100 partition provides quad-A100 SXM4 nodes. This
# implementation is single-GPU, so request one 80 GB A100 from the node.
#SBATCH --partition=swarm_a100
#SBATCH --gres=gpu:1
# The partition permits at most 120 hours (five days).
#SBATCH --time=5-00:00:00
#SBATCH --job-name=qwen253b-r1zero
# Submit from deepseek-R1-Zero so this relative path exists when Slurm opens it.
#SBATCH --output=result_out/qwen253b-r1zero-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/deepseek-R1-Zero"
MODEL_PATH="${HOME}/scratch/llms_model/GRPO-Zero/Qwen2.5-3B"
DATA_FILE="${PROJECT_ROOT}/rl_learning_demo/day07_GRPO/dapo/Countdown-Tasks-3to4/data/train-00000-of-00001.parquet"
OUTPUT_DIR="${HOME}/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-R1-Zero"
CONDA_ENV_NAME="rl_post_training_env"

# Make `conda activate` available inside a non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen253b-r1zero-${SLURM_JOB_ID}"

export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# Helps PyTorch reuse variably sized generation/training allocations.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_HOME}"
cd "${TRAINING_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Base model: ${MODEL_PATH}"
echo "Dataset: ${DATA_FILE}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "${TRAINING_DIR}/train.py" ]]; then
    echo "Training script missing: ${TRAINING_DIR}/train.py" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen2.5-3B Base checkpoint missing: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ "${MODEL_PATH,,}" == *instruct* ]]; then
    echo "R1-Zero-style training requires the Base model, not Instruct." >&2
    exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Countdown parquet file missing: ${DATA_FILE}" >&2
    exit 1
fi

"${PYTHON_EXECUTABLE}" --version
nvidia-smi
"${PYTHON_EXECUTABLE}" -c \
    "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'"

srun "${PYTHON_EXECUTABLE}" train.py \
    --model-path "${MODEL_PATH}" \
    --data-file "${DATA_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --question-batch-size 4 \
    --group-size 8 \
    --generation-batch-size 32 \
    --micro-batch-size 1 \
    --max-new-tokens 512 \
    --grpo-epochs 2 \
    --max-steps 1000 \
    --eval-every 50 \
    --save-every 50 \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved policy: ${OUTPUT_DIR}"

# cd ~/scratch/dips_project/reinforcement_learning/deepseek-R1-Zero
#
#   sbatch submit-R1-Zero-qwen253b.sh \
#       --max-steps 2 \
#       --save-every 2 \
#       --eval-every 2 \
#       --max-new-tokens 128
#
#  After that succeeds, submit the full run:
#
#  sbatch submit-R1-Zero-qwen253b.sh
#
#  The default output directory is:
#
#  ~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-R1-Zero
