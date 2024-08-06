#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=eth_stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/09_static_metrics.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/09_static_metrics.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@60

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/09_static_metrics.sh
