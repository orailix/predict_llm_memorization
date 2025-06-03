#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn perf --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn smi --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn p_smi --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn weights --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn logit_gap --config=DFmkqnNdCNzWI_i5AcwkYQ &
python -u -m grokking_llm measure-dyn sample_loss --config=DFmkqnNdCNzWI_i5AcwkYQ &

wait;
