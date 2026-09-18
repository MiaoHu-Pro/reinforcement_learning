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
#SBATCH --job-name=diff-unet
#SBATCH --output=result_out/diff-unet-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/multip_modal/diffusion-model"
DATA_DIR="${PROJECT_ROOT}/datasets/flickr8k/data"
OUTPUT_DIR="${TRAINING_DIR}/outputs/code-3-diff-unet-fixed"
CONDA_ENV_NAME="rl_post_training_env"

CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/diff-unet-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${TRAINING_DIR}"
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Flickr8k directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Arguments: $*"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -f "code-3-diff-unet-fixed.py" ]]; then
    echo "Training script is missing in ${TRAINING_DIR}" >&2
    exit 1
fi

python --version
python -c \
    "import datasets, PIL, torch, torchvision; print('datasets:', datasets.__version__); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('Pillow:', PIL.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
nvidia-smi

# Arguments after the shell script are forwarded to Python. Examples:
#   sbatch submit-code-3-diff-unet-fixed.sh
#   sbatch submit-code-3-diff-unet-fixed.sh --dataset flickr8k --epochs 100
srun python code-3-diff-unet-fixed.py \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

echo "Finished: $(date --iso-8601=seconds)"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "Final samples: ${OUTPUT_DIR}/final-samples.png"


# Submit Flickr8k training with:
 #
 #  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/diffusion-model
 #
#   sbatch submit-code-3-diff-unet-fixed.sh \
#       --dataset flickr8k \
#       --epochs 200 \
#       --sample-every 100 \
#       --sample-batch-size 16 \
#       --num-samples 256
 #
 #  Intermediate generated images will be saved at:
 #
 #  outputs/code-3-diff-unet-fixed/samples/epoch-0010.png
 #  outputs/code-3-diff-unet-fixed/samples/epoch-0020.png
 #  ...
 #
 #  Final outputs:
 #
 #  outputs/code-3-diff-unet-fixed/final-samples.png
 #  outputs/code-3-diff-unet-fixed/best.pt
 #  outputs/code-3-diff-unet-fixed/metrics.json
 #
 #  Copy an image back to your local computer with:
 #
 #  scp <server-host>:~/scratch/dips_project/reinforcement_learning/multip_modal/diffusion-model/outputs/code-3-diff-unet-fixed/samples/epoch-0010.png .
 #
 #  The demo mode remains available:
 #
 #  sbatch submit-code-3-diff-unet-fixed.sh
 #
 #  Both demo and Flickr8k end-to-end smoke tests passed, including training, validation/test loss, checkpoint saving, intermediate PNG creation, final sampling, and metrics output.


# > After every 10 training epochs, run the reverse diffusion process and save a grid of newly generated images.
  #
  #  For example, with:
  #
  #  --epochs 100 --sample-every 10
  #
  #  sample images are generated after epochs:
  #
  #  10, 20, 30, 40, 50, 60, 70, 80, 90, 100
  #
  #  They are saved as:
  #
  #  samples/epoch-0010.png
  #  samples/epoch-0020.png
  #  ...
  #  samples/epoch-0100.png
  #
  #  The number of generated images in each grid is controlled separately by:
  #
  #  --num-samples 16
  #
  #  Therefore:
  #
  #  --sample-every 10 --num-samples 16
  #
  #  means:
  #
  #  Every 10 epochs:
  #      start with 16 random-noise images
  #      perform the complete reverse-diffusion process
  #      generate 16 new images
  #      save them as one PNG grid
  #
  #  These generated images are not copies of Flickr8k images. They begin from random Gaussian noise:
  #
  #  x_T ~ N(0, I)
  #
  #  and are progressively denoised:
  #
  #  x_T → x_(T-1) → ... → x_1 → x_0
  #
  #  With the default:
  #
  #  --timesteps 1000
  #
  #  each sampling event performs:
  #
  #  16 images × 1000 denoising steps
  #
  #  The training images are handled independently. During each epoch, the DataLoader processes every Flickr8k training image approximately once:
  #
  #  6,000 training images / batch size 128
  #  ≈ 47 training batches per epoch
  #
  #  So:
  #
  #  - --epochs controls how many times the training dataset is traversed.
  #  - --sample-every controls how frequently generated examples are saved.
  #  - --num-samples controls how many images are generated at each sampling event.
  #  - --timesteps controls how many reverse-denoising steps generate each image.