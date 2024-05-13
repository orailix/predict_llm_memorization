#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=gemma_mmlu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_gemma/04_forward_target.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_gemma/04_forward_target.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@30

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_gemma/04_forward_target.sh
