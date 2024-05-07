#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=deploy_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:02:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/dummy_deploy/deploy_train.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/dummy_deploy/deploy_train.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/dummy_deploy/deploy_train.sh
