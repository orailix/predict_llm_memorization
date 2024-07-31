#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm measure-stat p_smi --config=66nIO9d4XxnsfDfc0jaExw --checkpoint=1200,15600 --njobs=40 --force-recompute
