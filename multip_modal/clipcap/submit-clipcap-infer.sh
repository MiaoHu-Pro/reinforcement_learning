#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=clipcap-infer
#SBATCH --output=result_out/clipcap-infer-%j.out

set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
CLIPCAP_DIR="${PROJECT_ROOT}/multip_modal/clipcap"
LLM_DIR="${HOME}/scratch/llms_model/gpt2-chinese-cluecorpussmall"
CLIP_DIR="${HOME}/scratch/llms_model/chinese-clip-vit-base-patch16"
CONDA_ENV_NAME="rl_post_training_env"

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
echo "Language model: ${LLM_DIR}"
echo "Chinese CLIP model: ${CLIP_DIR}"
echo "ClipCap checkpoint: ${CLIPCAP_DIR}/model.pt"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi
if [[ ! -d "${LLM_DIR}" ]]; then
    echo "Pretrained GPT-2 directory is missing: ${LLM_DIR}" >&2
    exit 1
fi
if [[ ! -d "${CLIP_DIR}" ]]; then
    echo "Pretrained Chinese CLIP directory is missing: ${CLIP_DIR}" >&2
    exit 1
fi
for required_file in infer.py model.py config.py model.pt trump.jpeg pokemon.jpeg; do
    if [[ ! -s "${required_file}" ]]; then
        echo "Required inference file is missing or empty: ${CLIPCAP_DIR}/${required_file}" >&2
        exit 1
    fi
done

python --version
python -c \
    "import PIL, torch, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Pillow:', PIL.__version__); assert torch.cuda.is_available(), 'Allocated GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

srun python infer.py

echo "Finished: $(date --iso-8601=seconds)"
