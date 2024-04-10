#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm measure-stat p_smi --config=66nIO9d4XxnsfDfc0jaExw --checkpoint=1200,15600