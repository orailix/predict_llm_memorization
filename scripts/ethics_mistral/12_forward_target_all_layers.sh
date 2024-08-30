#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=M-H-RqkPbMpWRGgOcg-PRA \
    --checkpoint=397,794,1191,1588,1985,3970,5955,7940,9925,11910,13895,15880,17865,19850 \
    --force-recompute;
