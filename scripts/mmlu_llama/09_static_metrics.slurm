#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=llama_mmlu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_llama/09_static_metrics.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_llama/09_static_metrics.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@60

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_llama/09_static_metrics.sh
