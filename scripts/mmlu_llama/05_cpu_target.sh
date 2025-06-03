#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn perf --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn smi --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn p_smi --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn weights --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn logit_gap --config=0K1pZkoAv45RZXOIJl9kFw &
python -u -m grokking_llm measure-dyn sample_loss --config=0K1pZkoAv45RZXOIJl9kFw &

wait;
