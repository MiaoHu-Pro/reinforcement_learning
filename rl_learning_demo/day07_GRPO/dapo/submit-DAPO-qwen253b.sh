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
#SBATCH --job-name=dapo-qwen253b
#SBATCH --output=result_out/dapo-qwen253b-%j.out

set -euo pipefail

# Ensure output directory exists before Slurm writes logs to it
mkdir -p result_out

CONDA_ENV_NAME="rl_post_training_env"

# Make `conda activate` available inside the non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"

# Keep temporary Hugging Face files on the compute node rather than in $HOME.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: ${PYTHON_EXECUTABLE}"

"${PYTHON_EXECUTABLE}" --version

nvidia-smi
"${PYTHON_EXECUTABLE}" -c \
    "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert torch.cuda.is_available(), 'PyTorch cannot access the allocated GPU'"

# Fixed: Added closing double quote
srun "${PYTHON_EXECUTABLE}" train.py

echo "Finished: $(date --iso-8601=seconds)"