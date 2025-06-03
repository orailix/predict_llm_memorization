#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn perf --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn smi --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn p_smi --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn weights --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn logit_gap --config=wd1yKu7ifTBhJncAHIe3vA &
python -u -m grokking_llm measure-dyn sample_loss --config=wd1yKu7ifTBhJncAHIe3vA &

wait;
