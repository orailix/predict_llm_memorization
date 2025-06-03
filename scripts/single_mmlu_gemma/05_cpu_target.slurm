#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=mml_g_ct
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_mmlu_gemma/05_cpu_target.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/single_mmlu_gemma/05_cpu_target.out
#SBATCH --no-requeue

# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_mmlu_gemma/05_cpu_target.sh
