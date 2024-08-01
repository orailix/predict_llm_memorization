#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/02_prepare_deploy.sh

for ((i=0; i<25; i++))
do
sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/03_deploy_train.slurm
done
