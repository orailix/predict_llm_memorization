#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward_on_all_layers \
    --config=Fp-lMrMPD6Br3DT_wCfPMA \
    --checkpoint=0,750,1500,2250,3000,3750 \
    --force-recompute;
