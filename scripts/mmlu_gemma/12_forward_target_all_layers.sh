#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=dRfdcYPXxygGJ2kuZfsYew \
    --checkpoint=0,750,1500,2250,3000,3750,7500,11250,15000,18750,22500,26250,30000,33750,37500 \
    --force-recompute;
