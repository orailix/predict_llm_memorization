#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=ethics_12
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/12_forward_target_all_layers.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/12_forward_target_all_layers.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@30

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/12_forward_target_all_layers.sh
