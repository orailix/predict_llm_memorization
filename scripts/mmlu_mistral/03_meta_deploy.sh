#!/bin/bash

bash /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/02_prepare_deploy.sh

for ((i=0; i<25; i++))
do
sbatch /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/03_deploy_train.slurm
done
