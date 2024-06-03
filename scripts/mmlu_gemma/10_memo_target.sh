#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn memo_on_shadow_mD9I --config=dRfdcYPXxygGJ2kuZfsYew &

wait;
