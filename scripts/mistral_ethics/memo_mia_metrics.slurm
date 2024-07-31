#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=memo_mia_metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/memo_mia_metrics.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/memo_mia_metrics.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/memo_mia_metrics.sh
