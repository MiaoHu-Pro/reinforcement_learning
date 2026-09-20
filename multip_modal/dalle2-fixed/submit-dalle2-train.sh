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
#SBATCH --job-name=dalle2-train
#SBATCH --output=result_out/dalle2-train-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
DALLE2_DIR="${PROJECT_ROOT}/multip_modal/dalle2-fixed"
DEFAULT_DATA_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
CONDA_ENV_NAME="rl_post_training_env"

# Flickr8k is the default for this new workflow. Arguments appended to sbatch
# are forwarded to all three Python stages and therefore can override these
# defaults, for example: --dataset fashion_mnist.
DATASET="flickr8k"
DATA_DIR="${DEFAULT_DATA_DIR}"
DATA_DIR_WAS_SET=false
USER_ARGS=("$@")
for ((index = 0; index < ${#USER_ARGS[@]}; index++)); do
    case "${USER_ARGS[index]}" in
        --dataset)
            DATASET="${USER_ARGS[index + 1]}"
            ;;
        --dataset=*)
            DATASET="${USER_ARGS[index]#*=}"
            ;;
        --data-dir)
            DATA_DIR="${USER_ARGS[index + 1]}"
            DATA_DIR_WAS_SET=true
            ;;
        --data-dir=*)
            DATA_DIR="${USER_ARGS[index]#*=}"
            DATA_DIR_WAS_SET=true
            ;;
    esac
done

if [[ "${DATASET}" == "fashion_mnist" ]]; then
    CHECKPOINT_SUFFIX="fmnist_fixed"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DALLE2_DIR}/datasets"
    fi
elif [[ "${DATASET}" == "flickr8k" ]]; then
    CHECKPOINT_SUFFIX="flickr8k"
else
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 1
fi

# Make `conda activate` available in the non-interactive Slurm shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/dalle2-${SLURM_JOB_ID}"
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
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Dataset: ${DATASET}"
echo "Data directory: ${DATA_DIR}"
echo "Additional arguments: ${USER_ARGS[*]}"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
for required_file in \
    train_clip.py train_prior.py train_decoder.py dalle2_dataset.py; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${DALLE2_DIR}/${required_file}" >&2
        exit 1
    fi
done
if [[ "${DATASET}" == "flickr8k" ]]; then
    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "Flickr8k directory is missing: ${DATA_DIR}" >&2
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
fi

mkdir -p "${HF_HOME}" "${DALLE2_DIR}/trained_models"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import datasets, PIL, torch, torchvision; print('datasets:', datasets.__version__); print('Pillow:', PIL.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# The stages are deliberately sequential: prior loads the trained CLIP, and
# decoder loads the trained prior. `set -e` stops immediately if a stage fails.
COMMON_ARGS=(--dataset "${DATASET}" --data-dir "${DATA_DIR}" --device cuda)

echo "========== Stage 1/3: CLIP =========="
srun "${PYTHON_EXECUTABLE}" train_clip.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"

echo "========== Stage 2/3: diffusion prior =========="
srun "${PYTHON_EXECUTABLE}" train_prior.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"

echo "========== Stage 3/3: diffusion decoder =========="
srun "${PYTHON_EXECUTABLE}" train_decoder.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"

for stage in clip prior decoder; do
    checkpoint="${DALLE2_DIR}/trained_models/${stage}_${CHECKPOINT_SUFFIX}.pt"
    if [[ ! -s "${checkpoint}" ]]; then
        echo "Training did not produce a non-empty checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    echo "Saved checkpoint: ${checkpoint}"
done

echo "Finished: $(date --iso-8601=seconds)"



#
#Created both executable Slurm scripts:
#
#  - multip_modal/dalle2-fixed/submit-dalle2-train.sh
#  - multip_modal/dalle2-fixed/submit-dalle2-infer.sh
#
#  The training job uses one 80 GB A100 from swarm_a100 for up to 120 hours and sequentially trains:
#
#  1. CLIP
#  2. Diffusion prior
#  3. Diffusion decoder
#
#  Submit training:
#
#  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/dalle2-fixed
#
#  sbatch submit-dalle2-train.sh
#
#  Flickr8k is the default dataset. To use FashionMNIST:
#
#  sbatch submit-dalle2-train.sh --dataset fashion_mnist
#
#  Submit inference after training succeeds:
#
#  TRAIN_JOB_ID=$(sbatch --parsable submit-dalle2-train.sh)
#
#  sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
#      submit-dalle2-infer.sh \
#      --prompt "a dog running through green grass" \
#      --num-images 4
#
#  The default inference output is:
#
#  multip_modal/dalle2-fixed/generated_images/dalle2-flickr8k.png
#
#  You can override it:
#
#  sbatch submit-dalle2-infer.sh \
#      --prompt "two dogs playing beside a lake" \
#      --num-images 8 \
#      --output generated_images/two-dogs.png
#