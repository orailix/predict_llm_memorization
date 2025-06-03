#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=p_tuSjzwdy_lf23W26F0Ww &
python -u -m grokking_llm measure-dyn perf --config=p_tuSjzwdy_lf23W26F0Ww &
python -u -m grokking_llm measure-dyn smi --config=p_tuSjzwdy_lf23W26F0Ww &
# python -u -m grokking_llm measure-dyn p_smi --config=p_tuSjzwdy_lf23W26F0Ww &
python -u -m grokking_llm measure-dyn weights --config=p_tuSjzwdy_lf23W26F0Ww &
python -u -m grokking_llm measure-dyn logit_gap --config=p_tuSjzwdy_lf23W26F0Ww &
python -u -m grokking_llm measure-dyn sample_loss --config=p_tuSjzwdy_lf23W26F0Ww &

wait;
