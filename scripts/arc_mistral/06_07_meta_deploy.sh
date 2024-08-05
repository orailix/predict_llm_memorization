#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/arc_mistral/02_prepare_deploy.sh

for ((i=0; i<10; i++)) # 7 jobs is enough for the 20h timeout but 10 is faster
do
sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/arc_mistral/06_07_forward_deploy.slurm
done
