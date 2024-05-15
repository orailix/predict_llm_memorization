#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=mistral_mmlu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_mistral/05_cpu_target.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_mistral/05_cpu_target.out
#SBATCH --no-requeue

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/05_cpu_target.sh
