#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/compress.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/compress.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Prepare deploy
bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/_prepare_deploy.sh

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/compress.sh
