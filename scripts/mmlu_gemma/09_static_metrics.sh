#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

python -u -m grokking_llm measure-stat memo_mia --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap_std --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute &
python -u -m grokking_llm measure-stat memo_counterfactual --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute &
python -u -m grokking_llm measure-stat simplicity_counterfactual --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute &
python -u -m grokking_llm measure-stat loss --config=mD9ImObzoCshT5dpStKiMA --checkpoint=750,1500 --force-recompute;

wait;
