#!/bin/bash

bash /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_gemma/02_prepare_deploy.sh

for ((i=0; i<3; i++))
do
sbatch /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_gemma/06_forward_deploy_compressed.slurm
done
