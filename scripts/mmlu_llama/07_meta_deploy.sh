#!/bin/bash

bash /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_llama/02_prepare_deploy.sh

for ((i=0; i<7; i++))
do
sbatch /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_llama/07_forward_deploy.slurm
done
