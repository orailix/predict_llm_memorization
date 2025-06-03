#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=1

# Prepare deploy
python -u -m grokking_llm deploy-prepare --config=JNeyT5OUP3GFtJ_QNojppg
