#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=p_tuSjzwdy_lf23W26F0Ww \
    --checkpoint=0,397,794,1191,1588,1985 \
    --force-recompute;
