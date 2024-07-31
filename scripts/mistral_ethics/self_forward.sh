#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm deploy-gpu --config=66nIO9d4XxnsfDfc0jaExw --no-training --self-forward=1200,15600 --self-forward-full-dataset
