#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=static_metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/static_metrics.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/static_metrics.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/static_metrics.sh
