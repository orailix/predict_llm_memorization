#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn perf --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn smi --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn p_smi --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn weights --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn logit_gap --config=oDw7PFxCTvAiAfSO86wWJA &
python -u -m grokking_llm measure-dyn sample_loss --config=oDw7PFxCTvAiAfSO86wWJA &

wait;
