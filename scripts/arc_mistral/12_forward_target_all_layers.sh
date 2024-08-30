#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=wd1yKu7ifTBhJncAHIe3vA \
    --checkpoint=0,50,100,150,200,250,500,750,1000,1250,1500,1750,2000,2250,2500 \
    --force-recompute;
