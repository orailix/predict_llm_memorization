#!/bin/bash

for ((i=0; i<2; i++))
do
sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/dummy_deploy/03_deploy_train.slurm
done
