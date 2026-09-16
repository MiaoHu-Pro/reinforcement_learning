#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=mnist-clip-fixed
# Submit from multip_modal/vit_and_clip so this relative directory exists.
#SBATCH --output=result_out/mnist-clip-fixed-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/multip_modal/vit_and_clip"
DATA_DIR="${TRAINING_DIR}/clip-mnist/mnist"
OUTPUT_FILE="${TRAINING_DIR}/checkpoints/code-2-clip-fixed-best.pt"
CONDA_ENV_NAME="rl_post_training_env"

# Make `conda activate` available in the non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/mnist-clip-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${TRAINING_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Dataset: ${DATA_DIR}"
echo "Checkpoint: ${OUTPUT_FILE}"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "${TRAINING_DIR}/code-2-clip-fixed.py" ]]; then
    echo "Training script missing: ${TRAINING_DIR}/code-2-clip-fixed.py" >&2
    exit 1
fi
if [[ ! -f "${DATA_DIR}/train-00000-of-00001.parquet" ]]; then
    echo "Training parquet missing from ${DATA_DIR}" >&2
    exit 1
fi
if [[ ! -f "${DATA_DIR}/test-00000-of-00001.parquet" ]]; then
    echo "Test parquet missing from ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${HF_HOME}" "$(dirname "${OUTPUT_FILE}")"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import datasets, torch, torchvision; print('datasets:', datasets.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

# Additional options are forwarded, for example:
# sbatch submit-code-2-clip-fixed.sh --epochs 1
srun "${PYTHON_EXECUTABLE}" code-2-clip-fixed.py \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_FILE}" \
    --num-workers 4 \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved checkpoint: ${OUTPUT_FILE}"

#  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/vit_and_clip
 #
 #  sbatch submit-code-2-clip-fixed.sh --epochs 1
 #
 #  Monitor it:
 #
 #  squeue -u "$USER"
 #  tail -f result_out/mnist-clip-fixed-<JOB_ID>.out
 #
 #  Then run the complete 20-epoch training:
 #
 #  sbatch submit-code-2-clip-fixed.sh