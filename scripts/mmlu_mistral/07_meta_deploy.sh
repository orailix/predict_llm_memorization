#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/02_prepare_deploy.sh

for ((i=0; i<7; i++))
do
sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/07_forward_deploy.slurm
done
