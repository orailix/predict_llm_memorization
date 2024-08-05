#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=eth_m_fc
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/06_forward_deploy_compressed.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/ethics_mistral/06_forward_deploy_compressed.out
#SBATCH --no-requeue
#SBATCH --signal=USR1@60

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/06_forward_deploy_compressed.sh
