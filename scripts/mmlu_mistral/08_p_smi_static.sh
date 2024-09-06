#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=40

# P-SMI dynamic
bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_mistral/02_prepare_deploy.sh
python -u -m grokking_llm deploy-cpu --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750 --only-metrics=p_smi_on_full_dataset --njobs=15;
python -u -m grokking_llm measure-stat p_smi --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750 --njobs=34 --force-recompute;