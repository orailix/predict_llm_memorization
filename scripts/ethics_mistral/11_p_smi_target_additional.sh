#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=13;

# CPU computations
python -u -m grokking_llm measure-dyn p_smi --config=M-H-RqkPbMpWRGgOcg-PRA &
python -u -m grokking_llm measure-dyn p_smi_slope --config=M-H-RqkPbMpWRGgOcg-PRA &
python -u -m grokking_llm measure-dyn p_smi_std --config=M-H-RqkPbMpWRGgOcg-PRA;

wait;
