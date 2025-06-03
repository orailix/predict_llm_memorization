#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/02_prepare_deploy.sh

for ((i=0; i<25; i++)) # 7 jobs is enough for the 20h timeout but 25 is faster
do
sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/06_07_forward_deploy.slurm
done
