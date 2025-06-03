#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_arc_llama/02_prepare_deploy.sh

sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_arc_llama/03_deploy_train.slurm
