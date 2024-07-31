#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=03_deploy_train
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:02:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/dummy_deploy/03_deploy_train.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/dummy_deploy/03_deploy_train.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@30

# Cleaning modules
module purge

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/dummy_deploy/03_deploy_train.sh
