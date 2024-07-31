#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn perf --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn smi --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn p_smi --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn weights --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn logit_gap --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn sample_loss --config=dRfdcYPXxygGJ2kuZfsYew &

wait;
