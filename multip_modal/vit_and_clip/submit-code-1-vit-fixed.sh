#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
# MNIST ViT is small; two hours includes environment startup and data checks.
#SBATCH --time=02:00:00
#SBATCH --job-name=mnist-vit-fixed
# Submit from multip_modal/vit_and_clip so this relative directory is resolved.
#SBATCH --output=result_out/mnist-vit-fixed-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/multip_modal/vit_and_clip"
DATA_DIR="${TRAINING_DIR}/datasets"
OUTPUT_FILE="${TRAINING_DIR}/checkpoints/code-1-vit-fixed-best.pt"
CONDA_ENV_NAME="rl_post_training_env"

# A non-interactive Slurm shell does not define the `conda activate` function
# until conda.sh has been sourced.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
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
if [[ ! -f "${TRAINING_DIR}/code-1-vit-fixed.py" ]]; then
    echo "Training script is missing: ${TRAINING_DIR}/code-1-vit-fixed.py" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}/MNIST/raw" ]]; then
    echo "Local MNIST files are missing: ${DATA_DIR}/MNIST/raw" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

# Extra arguments supplied after the Slurm filename are forwarded to Python.
# For example: sbatch submit-code-1-vit-fixed.sh --epochs 1
srun "${PYTHON_EXECUTABLE}" code-1-vit-fixed.py \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_FILE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --no-download \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved checkpoint: ${OUTPUT_FILE}"

#Run a one-epoch server test first:
 #
 #  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/vit_and_clip
 #
 #  sbatch submit-code-1-vit-fixed.sh --epochs 1
 #
 #  Monitor it with:
 #
 #  squeue -u "$USER"
 #  tail -f result_out/mnist-vit-fixed-<JOB_ID>.out
 #
 #  Then run the complete 15-epoch training:
 #
 #  sbatch submit-code-1-vit-fixed.sh
 #
 #  The best checkpoint and summary will be saved as:
 #
 #  multip_modal/vit_and_clip/checkpoints/code-1-vit-fixed-best.pt
 #  multip_modal/vit_and_clip/checkpoints/code-1-vit-fixed-best.json