#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=1

# Prepare deploy
python -u -m grokking_llm deploy-prepare --config=Z5n7bEDGK4JRT4HyLKSzrw
