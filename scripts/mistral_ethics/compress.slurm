#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/compress.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/compress.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/compress.sh
