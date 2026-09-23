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
DEFAULT_FLICKR8K_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
DEFAULT_FLICKR30K_DIR="${PROJECT_ROOT}/datasets/flickr30k/data"
DEFAULT_ALL_DATA_DIR="${PROJECT_ROOT}/datasets"
DEFAULT_PRETRAINED_CLIP_DIR="${HOME}/scratch/llms_model/clip-vit-base-patch32"
CONDA_ENV_NAME="rl_post_training_env"

# Flickr30k is the default for this workflow. Arguments appended to sbatch
# are forwarded to all three Python stages and therefore can override these
# defaults, for example: --dataset all or --dataset flickr8k.
DATASET="flickr30k"
DATA_DIR="${DEFAULT_FLICKR30K_DIR}"
DATA_DIR_WAS_SET=false
USING_PRETRAINED_CLIP=false
PRETRAINED_CLIP_DIR="${DEFAULT_PRETRAINED_CLIP_DIR}"
LARGE_UNET=false
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
    CHECKPOINT_SUFFIX="fmnist_fixed"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DALLE2_DIR}/datasets"
    fi
elif [[ "${DATASET}" == "flickr8k" ]]; then
    CHECKPOINT_SUFFIX="flickr8k"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_FLICKR8K_DIR}"
    fi
elif [[ "${DATASET}" == "flickr30k" ]]; then
    CHECKPOINT_SUFFIX="flickr30k"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_FLICKR30K_DIR}"
    fi
elif [[ "${DATASET}" == "all" ]]; then
    CHECKPOINT_SUFFIX="all"
    if [[ "${DATA_DIR_WAS_SET}" == false ]]; then
        DATA_DIR="${DEFAULT_ALL_DATA_DIR}"
    fi
else
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 1
fi
if [[ "${USING_PRETRAINED_CLIP}" == true ]]; then
    CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX}_preclip"
fi
DECODER_CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX}"
if [[ "${LARGE_UNET}" == true ]]; then
    DECODER_CHECKPOINT_SUFFIX="${DECODER_CHECKPOINT_SUFFIX}_largeunet"
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
echo "Using pretrained CLIP: ${USING_PRETRAINED_CLIP}"
echo "Using large U-Net: ${LARGE_UNET}"
if [[ "${USING_PRETRAINED_CLIP}" == true ]]; then
    echo "Pretrained CLIP directory: ${PRETRAINED_CLIP_DIR}"
fi
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
check_flickr_parquet_directory() {
    local dataset_name="$1"
    local dataset_dir="$2"
    if [[ ! -d "${dataset_dir}" ]]; then
        echo "${dataset_name} directory is missing: ${dataset_dir}" >&2
        exit 1
    fi
    if ! compgen -G "${dataset_dir}/*.parquet" > /dev/null; then
        echo "${dataset_name} parquet files are missing from ${dataset_dir}" >&2
        exit 1
    fi
}

if [[ "${DATASET}" == "flickr8k" || "${DATASET}" == "flickr30k" ]]; then
    check_flickr_parquet_directory "${DATASET}" "${DATA_DIR}"
elif [[ "${DATASET}" == "all" ]]; then
    check_flickr_parquet_directory "Flickr8k" "${DATA_DIR}/flickr8k/data"
    check_flickr_parquet_directory "Flickr30k" "${DATA_DIR}/flickr30k/data"
fi
if [[ "${USING_PRETRAINED_CLIP}" == true && ! -d "${PRETRAINED_CLIP_DIR}" ]]; then
    echo "Pretrained CLIP directory is missing: ${PRETRAINED_CLIP_DIR}" >&2
    exit 1
fi

mkdir -p "${HF_HOME}" "${DALLE2_DIR}/trained_models"

"${PYTHON_EXECUTABLE}" --version
"${PYTHON_EXECUTABLE}" -c \
    "import datasets, PIL, torch, torchvision; print('datasets:', datasets.__version__); print('Pillow:', PIL.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# The stages are deliberately sequential: prior loads the trained CLIP, and
# decoder loads the trained prior. `set -e` stops immediately if a stage fails.
COMMON_ARGS=(--dataset "${DATASET}" --device cuda)

if [[ "${USING_PRETRAINED_CLIP}" == true ]]; then
    echo "========== Stage 1/3: frozen pretrained CLIP (training skipped) =========="
else
    echo "========== Stage 1/3: train custom CLIP =========="
    srun "${PYTHON_EXECUTABLE}" train_clip.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"
fi

echo "========== Stage 2/3: diffusion prior =========="
srun "${PYTHON_EXECUTABLE}" train_prior.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"

echo "========== Stage 3/3: diffusion decoder =========="
srun "${PYTHON_EXECUTABLE}" train_decoder.py "${COMMON_ARGS[@]}" "${USER_ARGS[@]}"

CHECKPOINTS=(
    "${DALLE2_DIR}/trained_models/prior_${CHECKPOINT_SUFFIX}.pt"
    "${DALLE2_DIR}/trained_models/decoder_${DECODER_CHECKPOINT_SUFFIX}.pt"
)
if [[ "${USING_PRETRAINED_CLIP}" == false ]]; then
    CHECKPOINTS=(
        "${DALLE2_DIR}/trained_models/clip_${CHECKPOINT_SUFFIX}.pt"
        "${CHECKPOINTS[@]}"
    )
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
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
#  Flickr30k is the default dataset. To combine both Flickr datasets:
#
#  sbatch submit-dalle2-train.sh --dataset all
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
#  multip_modal/dalle2-fixed/generated_images/dalle2-flickr30k.png
#
#  You can override it:
#
#  sbatch submit-dalle2-infer.sh \
#      --prompt "two dogs playing beside a lake" \
#      --num-images 8 \
#      --output generated_images/two-dogs.png
#
