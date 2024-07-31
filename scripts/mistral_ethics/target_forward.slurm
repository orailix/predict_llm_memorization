#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=target_forward
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/target_forward.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/target_forward.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/target_forward.sh
