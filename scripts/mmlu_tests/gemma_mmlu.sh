#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Training
python -u /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/setup.py;
python -u -m grokking_llm train --config=7sLH0svuW_pNh8Uecl5Exg;

# Basic measures
python -u -m grokking_llm measure-dyn general --config=7sLH0svuW_pNh8Uecl5Exg ;
python -u -m grokking_llm measure-dyn perf --config=7sLH0svuW_pNh8Uecl5Exg;

# Forward
python -u -m grokking_llm measure-dyn forward --config=7sLH0svuW_pNh8Uecl5Exg;

# Other metrics
python -u -m grokking_llm measure-dyn smi --config=7sLH0svuW_pNh8Uecl5Exg &
python -u -m grokking_llm measure-dyn p_smi --config=7sLH0svuW_pNh8Uecl5Exg &
python -u -m grokking_llm measure-dyn weights --config=7sLH0svuW_pNh8Uecl5Exg &
python -u -m grokking_llm measure-dyn memo_proba_gap --config=7sLH0svuW_pNh8Uecl5Exg &
python -u -m grokking_llm measure-dyn logit_gap --config=7sLH0svuW_pNh8Uecl5Exg &
python -u -m grokking_llm measure-dyn sample_loss --config=7sLH0svuW_pNh8Uecl5Exg;
wait;
