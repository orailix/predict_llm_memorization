#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -m grokking_llm deploy-cpu --config=66nIO9d4XxnsfDfc0jaExw --only-metrics=compress_forward --njobs=10
