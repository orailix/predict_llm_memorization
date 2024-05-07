#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=mistral_mmlu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_mistral/03_deploy_train.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/mmlu_mistral/03_deploy_train.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@30

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/03_deploy_train.sh
