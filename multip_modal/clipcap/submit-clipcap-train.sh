#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=clipcap-train
#SBATCH --output=result_out/clipcap-train-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
CLIPCAP_DIR="${PROJECT_ROOT}/multip_modal/clipcap"
LLM_DIR="${HOME}/scratch/llms_model/gpt2-chinese-cluecorpussmall"
CONDA_ENV_NAME="rl_post_training_env"

# `conda activate` is not initialized automatically in a Slurm batch shell.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${CLIPCAP_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Pretrained language model: ${LLM_DIR}"
echo "Output checkpoint: ${CLIPCAP_DIR}/model.pt"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -d "${LLM_DIR}" ]]; then
    echo "Pretrained GPT-2 directory is missing: ${LLM_DIR}" >&2
    exit 1
fi
for required_file in train.py model.py config.py clipcap_dataset.py caption_image.pkl; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required ClipCap file is missing: ${CLIPCAP_DIR}/${required_file}" >&2
        exit 1
    fi
done

python --version
python -c \
    "import torch, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

# train.py writes model.pt in the current directory.  The cd above ensures that
# infer.py will later read exactly the checkpoint produced by this job.
srun python train.py

if [[ ! -s "model.pt" ]]; then
    echo "Training finished without producing a non-empty model.pt" >&2
    exit 1
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "Saved checkpoint: ${CLIPCAP_DIR}/model.pt"


# Submit training followed automatically by inference:
  #
  #  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/clipcap
  #
  #  TRAIN_JOB_ID=$(sbatch --parsable submit-clipcap-train.sh)
  #  sbatch --dependency="afterok:${TRAIN_JOB_ID}" submit-clipcap-infer.sh
  #
  #  The afterok dependency means inference starts only if training succeeds.
  #
  #  The paths currently expected are:
  #
  #  ~/scratch/llms_model/gpt2-chinese-cluecorpussmall
  #  ~/scratch/llms_model/chinese-clip-vit-base-patch16
  #
  #  Training writes:
  #
  #  multip_modal/clipcap/model.pt
  #
  #  Inference reads that same checkpoint.
  #
  #  The GPT-2 path exists on the current machine. The Chinese-CLIP path does not exist locally, so ensure it exists on the server:
  #
  #  ls ~/scratch/llms_model/chinese-clip-vit-base-patch16
  #
  #  The inference job performs this check before loading the model.