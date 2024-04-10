#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=cpu_target_metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/cpu_target_metrics.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/cpu_target_metrics.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/cpu_target_metrics.sh
