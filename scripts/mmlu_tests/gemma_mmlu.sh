#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Training
python -u /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/setup.py;
python -u -m grokking_llm train --config=dRfdcYPXxygGJ2kuZfsYew;

# Basic measures
python -u -m grokking_llm measure-dyn general --config=dRfdcYPXxygGJ2kuZfsYew;

# Forward
python -u -m grokking_llm measure-dyn forward --config=dRfdcYPXxygGJ2kuZfsYew;

# Other metrics
python -u -m grokking_llm measure-dyn perf --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn smi --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn p_smi --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn weights --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn logit_gap --config=dRfdcYPXxygGJ2kuZfsYew &
python -u -m grokking_llm measure-dyn sample_loss --config=dRfdcYPXxygGJ2kuZfsYew;
wait;
