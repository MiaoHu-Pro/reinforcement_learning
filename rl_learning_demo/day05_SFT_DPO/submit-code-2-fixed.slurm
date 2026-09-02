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
#SBATCH --job-name=qwen3-0.6b-sft
# Slurm does not expand ~, $HOME, or shell variables in #SBATCH directives.
# Submit this file from day05_SFT_DPO so this relative path resolves correctly.
#SBATCH --output=result_out/qwen3-0.6b-sft-%j.out

# Stop immediately when a command fails, an undefined variable is used, or a
# command inside a pipeline fails. This prevents a failed setup from being
# mistaken for a successful training job.
set -euo pipefail

PROJECT_ROOT="${HOME}/scratch/dips_project/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/rl_learning_demo/day05_SFT_DPO"
MODEL_PATH="${HOME}/scratch/llms_model/Qwen3-0.6B"
CONDA_ENV_NAME="rl_post_training_env"

# Non-interactive Slurm shells do not automatically define `conda activate`.
# Load Conda's shell integration first, then activate the server environment.
# Temporarily disabling nounset avoids activation scripts failing on optional
# interactive-shell variables such as PS1.
CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

PYTHON_EXECUTABLE="$(command -v python)"

# Use local project/model/data files. The temporary Hugging Face cache prevents
# concurrent Slurm jobs from competing for a shared cache lock.
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen3-sft-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${HF_HOME}"
cd "${TRAINING_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Model: ${MODEL_PATH}"
"${PYTHON_EXECUTABLE}" --version
echo "Started: $(date --iso-8601=seconds)"

if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    echo "Expected Conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV}" >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

# Record the assigned GPU and its available memory in the Slurm output file.
nvidia-smi

# srun launches the training process inside the resources allocated by sbatch.
# code-2-fixed.py currently defaults to 200,000 samples, batch size 2,
# and one epoch.
srun "${PYTHON_EXECUTABLE}" code-2-fixed.py \
    --model-path "${MODEL_PATH}" \
    --device cuda

echo "Finished: $(date --iso-8601=seconds)"

#python -m pip install \
#      "gymnasium>=1.0" \
#      "numpy==1.26" \
#      "matplotlib>=3.11.1" \
#      "pygame>=2.6.1" \
#      "torch>=2.7" \
#      "transformers[serving]>=5.16.1" \
#      "datasets>=5.0.1" \
#      "modelscope>=1.39.1"
