#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# P-SMI dynamic
bash /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_gemma/02_prepare_deploy.sh
python -u -m grokking_llm deploy-cpu --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --only-metrics=p_smi_on_full_dataset --njobs=40;
python -u -m grokking_llm measure-stat p_smi --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --njobs=40 --force-recompute;
