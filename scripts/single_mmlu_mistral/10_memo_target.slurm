#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=m_m_mt
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_mmlu_mistral/10_memo_target%a.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_mmlu_mistral/10_memo_target%a.out
#SBATCH --no-requeue

# Code execution with binding via bind_gpu.sh : 1 GPU per task
all_checkpoints=("0" "750" "1500" "2250" "3000" "3750")
export checkpoint="${all_checkpoints[$SLURM_ARRAY_TASK_ID]}"
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_mmlu_mistral/10_memo_target.sh
