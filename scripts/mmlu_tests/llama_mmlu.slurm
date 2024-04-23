#!/bin/bash

#SBATCH --account=yfw@a100
#SBATCH --job-name=llama_mmlu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/llama_mmlu.out
#SBATCH --error=/gpfswork/rech/yfw/upp42qa/grokking_llm/logs/llama_mmlu.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Setup
bash /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/setup.sh

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/bind_gpu.sh /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/llama_mmlu.sh
