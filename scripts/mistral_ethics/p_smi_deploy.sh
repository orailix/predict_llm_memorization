#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm deploy-cpu --config=66nIO9d4XxnsfDfc0jaExw --only-metrics=p_smi --njobs=10 --checkpoint=1200
python -u -m grokking_llm deploy-cpu --config=66nIO9d4XxnsfDfc0jaExw --only-metrics=p_smi --njobs=10 --checkpoint=15600
