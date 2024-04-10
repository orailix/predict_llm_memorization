#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=clean_forward
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prepost
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/clean_forward.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/clean_forward.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/clean_forward.sh
