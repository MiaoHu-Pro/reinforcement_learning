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
#SBATCH --output=/local/scratch/mh1f25/python_projects/reinforcement_learning/rl_learning_demo/day05_SFT_DPO/result_out/qwen3-0.6b-sft-%j.out

# Stop immediately when a command fails, an undefined variable is used, or a
# command inside a pipeline fails. This prevents a failed setup from being
# mistaken for a successful training job.
set -euo pipefail

PROJECT_ROOT="/local/scratch/mh1f25/python_projects/reinforcement_learning"
TRAINING_DIR="${PROJECT_ROOT}/rl_learning_demo/day05_SFT_DPO"
PYTHON_EXECUTABLE="${PROJECT_ROOT}/.venv/bin/python"

# Use local project/model/data files. The temporary Hugging Face cache prevents
# concurrent Slurm jobs from competing for a shared cache lock.
JOB_TEMP_DIR="${SLURM_TMPDIR:-/tmp}/qwen3-sft-${SLURM_JOB_ID}"
export HF_HOME="${JOB_TEMP_DIR}/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${HF_HOME}"
cd "${TRAINING_DIR}"

echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: ${PYTHON_EXECUTABLE}"
echo "Started: $(date --iso-8601=seconds)"

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
    echo "Python environment not found: ${PYTHON_EXECUTABLE}" >&2
    exit 1
fi

# Record the assigned GPU and its available memory in the Slurm output file.
nvidia-smi

# srun launches the training process inside the resources allocated by sbatch.
# code-2-fixed.py defaults to 5,000 samples, batch size 2, and one epoch.
srun "${PYTHON_EXECUTABLE}" code-2-fixed.py --device cuda

echo "Finished: $(date --iso-8601=seconds)"
