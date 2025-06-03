#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=ZauochgWxZxlxg2eGfIXCw \
    --checkpoint=0,50,100,150,200,250 \
    --force-recompute;
