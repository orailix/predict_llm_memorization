#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=ethics_g_fc
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --array=0-25
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_ethics_gemma/06_07_forward_deploy.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_ethics_gemma/06_07_forward_deploy.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@60

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_ethics_gemma/06_07_forward_deploy.sh
