#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=mistral_arc
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/03_deploy_train.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/03_deploy_train.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@30

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/03_deploy_train.sh
