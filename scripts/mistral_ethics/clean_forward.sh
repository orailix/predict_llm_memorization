#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
# Use this to clean only compressed metrics:
python -m grokking_llm deploy-clean-forward-values --config=66nIO9d4XxnsfDfc0jaExw --compressed-only
# And that to clean all metrics:
# python -m grokking_llm deploy-clean-forward-values --config=66nIO9d4XxnsfDfc0jaExw --no-compressed-only
