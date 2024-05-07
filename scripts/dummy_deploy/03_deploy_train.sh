#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Prepare deploy
python -u -m grokking_llm deploy-gpu \
    --config=Nu00vbIUZIlYkpR_g0LnwQ \
    --training;
