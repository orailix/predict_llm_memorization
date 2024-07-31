#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm deploy-cpu --config=66nIO9d4XxnsfDfc0jaExw --only-metrics=compress_forward --njobs=10
