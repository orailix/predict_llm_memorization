#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=p_smi_static
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/p_smi_static.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/p_smi_static.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/p_smi_static.sh
