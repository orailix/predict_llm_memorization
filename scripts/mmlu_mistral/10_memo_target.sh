#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn memo_on_shadow_Z5n7 --config=Ji5OIUTJGuYCYhAzj6KtZQ &

wait;
