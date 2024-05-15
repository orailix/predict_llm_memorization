#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=8

# Measure forward
python -u -m grokking_llm measure-dyn forward --config=0K1pZkoAv45RZXOIJl9kFw;
