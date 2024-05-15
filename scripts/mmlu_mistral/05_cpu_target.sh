#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn perf --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn smi --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn p_smi --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn weights --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn logit_gap --config=Ji5OIUTJGuYCYhAzj6KtZQ &
python -u -m grokking_llm measure-dyn sample_loss --config=Ji5OIUTJGuYCYhAzj6KtZQ &

wait;
