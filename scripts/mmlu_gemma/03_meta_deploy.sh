#!/bin/bash

for ((i=0; i<25; i++))
do
sbatch /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_gemma/03_deploy_train.slurm
done
