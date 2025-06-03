#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=uXa6F4uHKeBcoexFs0nKlQ &
python -u -m grokking_llm measure-dyn perf --config=uXa6F4uHKeBcoexFs0nKlQ &
python -u -m grokking_llm measure-dyn smi --config=uXa6F4uHKeBcoexFs0nKlQ &
# python -u -m grokking_llm measure-dyn p_smi --config=uXa6F4uHKeBcoexFs0nKlQ &
python -u -m grokking_llm measure-dyn weights --config=uXa6F4uHKeBcoexFs0nKlQ &
python -u -m grokking_llm measure-dyn logit_gap --config=uXa6F4uHKeBcoexFs0nKlQ &
python -u -m grokking_llm measure-dyn sample_loss --config=uXa6F4uHKeBcoexFs0nKlQ &

wait;
