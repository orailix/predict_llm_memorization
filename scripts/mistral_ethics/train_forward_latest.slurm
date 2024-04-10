#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=train_forward_latest
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/train_forward_latest.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/train_forward_latest.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/train_forward_latest.sh
