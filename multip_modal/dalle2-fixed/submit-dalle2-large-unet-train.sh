#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --job-name=dalle2-large-unet
#SBATCH --output=result_out/dalle2-large-unet-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
DALLE2_DIR="${PROJECT_ROOT}/multip_modal/dalle2-fixed"
DATA_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
PRETRAINED_CLIP_DIR="${HOME}/scratch/llms_model/clip-vit-base-patch32"
PRIOR_CHECKPOINT="${DALLE2_DIR}/trained_models/prior_flickr8k_preclip.pt"
OUTPUT_CHECKPOINT="${DALLE2_DIR}/trained_models/decoder_flickr8k_preclip_largeunet.pt"
CONDA_ENV_NAME="rl_post_training_env"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/dalle2-large-unet-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${DALLE2_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Dataset: ${DATA_DIR}"
echo "Pretrained CLIP: ${PRETRAINED_CLIP_DIR}"
echo "Reusing prior: ${PRIOR_CHECKPOINT}"
echo "Large U-Net output: ${OUTPUT_CHECKPOINT}"
echo "Additional arguments: $*"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "train_decoder.py" ]]; then
    echo "Missing decoder training script: ${DALLE2_DIR}/train_decoder.py" >&2
    exit 1
fi
if [[ ! -d "${PRETRAINED_CLIP_DIR}" ]]; then
    echo "Missing pretrained CLIP: ${PRETRAINED_CLIP_DIR}" >&2
    exit 1
fi
if [[ ! -s "${PRIOR_CHECKPOINT}" ]]; then
    echo "Missing pretrained-CLIP prior checkpoint: ${PRIOR_CHECKPOINT}" >&2
    exit 1
fi
if ! compgen -G "${DATA_DIR}/train-*.parquet" > /dev/null; then
    echo "Flickr8k training parquet files are missing from ${DATA_DIR}" >&2
    exit 1
fi
if ! compgen -G "${DATA_DIR}/validation-*.parquet" > /dev/null; then
    echo "Flickr8k validation parquet files are missing from ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${HF_HOME}" "${DALLE2_DIR}/trained_models"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import datasets, torch, torchvision, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; name=torch.cuda.get_device_name(0); memory=torch.cuda.get_device_properties(0).total_memory/1024**3; print('GPU:', name); print(f'VRAM: {memory:.1f} GiB'); assert memory >= 70, 'This job expects an approximately 80 GB GPU'"
nvidia-smi

# Only the decoder is new. The frozen pretrained CLIP and its trained prior are
# loaded from disk, while --large-UNet selects the wider 64/128/256/512 U-Net.
srun "${PYTHON_EXECUTABLE}" train_decoder.py \
    --dataset flickr8k \
    --data-dir "${DATA_DIR}" \
    --device cuda \
    --using-pre-CLIP \
    --pretrained-clip-path "${PRETRAINED_CLIP_DIR}" \
    --large-UNet \
    "$@"

if [[ ! -s "${OUTPUT_CHECKPOINT}" ]]; then
    echo "Training did not produce: ${OUTPUT_CHECKPOINT}" >&2
    exit 1
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved large U-Net: ${OUTPUT_CHECKPOINT}"
