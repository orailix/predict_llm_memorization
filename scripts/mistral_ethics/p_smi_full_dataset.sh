#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm deploy-cpu --config=66nIO9d4XxnsfDfc0jaExw --checkpoint=1200,15600 --only-metrics=p_smi_on_full_dataset --njobs=20
