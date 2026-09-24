#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=dalle2-infer
#SBATCH --output=result_out/dalle2-infer-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
DALLE2_DIR="${PROJECT_ROOT}/multip_modal/dalle2-fixed"
DEFAULT_FLICKR8K_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
DEFAULT_FLICKR30K_DIR="${PROJECT_ROOT}/datasets/flickr30k/data"
DEFAULT_ALL_DATA_DIR="${PROJECT_ROOT}/datasets"
DEFAULT_PRETRAINED_CLIP_DIR="${HOME}/scratch/llms_model/clip-vit-base-patch32"
CONDA_ENV_NAME="rl_post_training_env"

DATASET="flickr30k"
DATA_DIR="${DEFAULT_FLICKR30K_DIR}"
DATA_DIR_WAS_SET=false
USING_PRETRAINED_CLIP=false
PRETRAINED_CLIP_DIR="${DEFAULT_PRETRAINED_CLIP_DIR}"
LARGE_UNET=false
RUN_NAME=""
USER_ARGS=("$@")
for ((index = 0; index < ${#USER_ARGS[@]}; index++)); do
    case "${USER_ARGS[index]}" in
        --dataset|--data)
            DATASET="${USER_ARGS[index + 1]}"
            ;;
        --dataset=*|--data=*)
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
        --using-pre-CLIP|--using-pre-clip)
            USING_PRETRAINED_CLIP=true
            ;;
        --pretrained-clip-path)
            PRETRAINED_CLIP_DIR="${USER_ARGS[index + 1]}"
            ;;
        --pretrained-clip-path=*)
            PRETRAINED_CLIP_DIR="${USER_ARGS[index]#*=}"
            ;;
        --large-UNet|--large-unet)
            LARGE_UNET=true
            ;;
        --run-name)
            RUN_NAME="${USER_ARGS[index + 1]}"
            ;;
        --run-name=*)
            RUN_NAME="${USER_ARGS[index]#*=}"
            ;;
    esac
done

# Match the aliases accepted by the Python argument parser.
case "${DATASET}" in
    flick8k)
        DATASET="flickr8k"
        ;;
    fashionMNIST|fashionmnist|fashion-mnist)
        DATASET="fashion_mnist"
        ;;
esac

if [[ "${DATASET}" == "fashion_mnist" ]]; then
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-fashion-mnist.png"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DALLE2_DIR}/datasets"
    fi
elif [[ "${DATASET}" == "flickr8k" ]]; then
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-flickr8k.png"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_FLICKR8K_DIR}"
    fi
elif [[ "${DATASET}" == "flickr30k" ]]; then
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-flickr30k.png"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_FLICKR30K_DIR}"
    fi
elif [[ "${DATASET}" == "all" ]]; then
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/dalle2-all.png"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_ALL_DATA_DIR}"
    fi
else
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 1
fi
if [[ -n "${RUN_NAME}" ]]; then
    DEFAULT_OUTPUT="${DALLE2_DIR}/generated_images/${RUN_NAME}.png"
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
if [[ -n "${RUN_NAME}" ]]; then
    echo "Experiment: ${RUN_NAME} (configuration loaded from manifest)"
else
    echo "Dataset: ${DATASET}"
    echo "Using pretrained CLIP: ${USING_PRETRAINED_CLIP}"
    echo "Using large U-Net: ${LARGE_UNET}"
fi
echo "Additional arguments: ${USER_ARGS[*]}"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "infer.py" || ! -f "config.py" ]]; then
    echo "Inference or configuration script is missing in ${DALLE2_DIR}" >&2
    exit 1
fi
if [[ "${USING_PRETRAINED_CLIP}" == true && ! -d "${PRETRAINED_CLIP_DIR}" ]]; then
    echo "Pretrained CLIP directory is missing: ${PRETRAINED_CLIP_DIR}" >&2
    exit 1
fi
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
    --device cuda \
    --output "${DEFAULT_OUTPUT}" \
    "${USER_ARGS[@]}"

echo "Finished: $(date --iso-8601=seconds)"
