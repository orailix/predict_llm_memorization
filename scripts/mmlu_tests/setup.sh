#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Training
python /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/setup.py
