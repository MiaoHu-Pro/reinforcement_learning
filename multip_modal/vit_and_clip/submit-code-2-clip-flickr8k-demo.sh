#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=swarm_a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=flickr8k-clip
# Submit from multip_modal/vit_and_clip so this relative directory exists.
#SBATCH --output=result_out/flickr8k-clip-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/multip_modal/vit_and_clip"
DATA_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
OUTPUT_FILE="${TRAINING_DIR}/checkpoints/code-2-clip-flickr8k-demo-best.pt"
CONDA_ENV_NAME="rl_post_training_env"

# Enable `conda activate` inside the non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/flickr8k-clip-${SLURM_JOB_ID}"
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
if [[ ! -f "${TRAINING_DIR}/code-2-clip-flickr8k-demo.py" ]]; then
    echo "Training script is missing in ${TRAINING_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Flickr8k data directory is missing: ${DATA_DIR}" >&2
    exit 1
fi
if ! compgen -G "${DATA_DIR}/train-*.parquet" > /dev/null; then
    echo "Flickr8k training parquet files are missing" >&2
    exit 1
fi
if ! compgen -G "${DATA_DIR}/validation-*.parquet" > /dev/null; then
    echo "Flickr8k validation parquet file is missing" >&2
    exit 1
fi
if ! compgen -G "${DATA_DIR}/test-*.parquet" > /dev/null; then
    echo "Flickr8k test parquet file is missing" >&2
    exit 1
fi

mkdir -p "${HF_HOME}" "$(dirname "${OUTPUT_FILE}")"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import datasets, torch, torchvision; print('datasets:', datasets.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# Extra arguments are forwarded to Python. Example smoke test:
# sbatch submit-code-2-clip-flickr8k-demo.sh --epochs 1 \
#   --max-train-images 512 --max-validation-images 100 --max-test-images 100
srun "${PYTHON_EXECUTABLE}" code-2-clip-flickr8k-demo.py \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT_FILE}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved checkpoint: ${OUTPUT_FILE}"
