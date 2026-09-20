#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=swarm_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=dalle2-infer
#SBATCH --output=result_out/dalle2-infer-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
DALLE2_DIR="${PROJECT_ROOT}/multip_modal/dalle2-fixed"
DEFAULT_DATA_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
CONDA_ENV_NAME="rl_post_training_env"

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
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-fashion-mnist.png"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DALLE2_DIR}/datasets"
    fi
elif [[ "${DATASET}" == "flickr8k" ]]; then
    CHECKPOINT_SUFFIX="flickr8k"
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-flickr8k.png"
else
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"
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
echo "Dataset: ${DATASET}"
echo "Additional arguments: ${USER_ARGS[*]}"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "infer.py" ]]; then
    echo "Inference script is missing: ${DALLE2_DIR}/infer.py" >&2
    exit 1
fi
for stage in clip prior decoder; do
    checkpoint="${DALLE2_DIR}/trained_models/${stage}_${CHECKPOINT_SUFFIX}.pt"
    if [[ ! -s "${checkpoint}" ]]; then
        echo "Required checkpoint is missing or empty: ${checkpoint}" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${DEFAULT_OUTPUT}")"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import PIL, torch, torchvision; print('Pillow:', PIL.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

# Extra arguments override repeated defaults because argparse keeps the final
# occurrence. Example:
# sbatch submit-dalle2-infer.sh --prompt "two dogs playing" \
#   --num-images 8 --output generated_images/two-dogs.png
srun "${PYTHON_EXECUTABLE}" infer.py \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --device cuda \
    --output "${DEFAULT_OUTPUT}" \
    "${USER_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"
