#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=ZauochgWxZxlxg2eGfIXCw &
python -u -m grokking_llm measure-dyn perf --config=ZauochgWxZxlxg2eGfIXCw &
python -u -m grokking_llm measure-dyn smi --config=ZauochgWxZxlxg2eGfIXCw &
# python -u -m grokking_llm measure-dyn p_smi --config=ZauochgWxZxlxg2eGfIXCw &
python -u -m grokking_llm measure-dyn weights --config=ZauochgWxZxlxg2eGfIXCw &
python -u -m grokking_llm measure-dyn logit_gap --config=ZauochgWxZxlxg2eGfIXCw &
python -u -m grokking_llm measure-dyn sample_loss --config=ZauochgWxZxlxg2eGfIXCw &

wait;
