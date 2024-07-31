#!/bin/bash

#SBATCH --account=yfw@cpu
#SBATCH --job-name=p_smi_full_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu_p1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/p_smi_full_dataset.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/logs/p_smi_full_dataset.out
#SBATCH --no-requeue

# Cleaning modules
module purge

# Prepare deploy
bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/_prepare_deploy.sh

# Code execution
srun /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mistral_ethics/p_smi_full_dataset.sh
