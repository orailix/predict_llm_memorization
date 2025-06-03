#!/bin/bash

bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_arc_gemma/02_prepare_deploy.sh

sbatch /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/single_arc_gemma/06_07_forward_deploy.slurm
