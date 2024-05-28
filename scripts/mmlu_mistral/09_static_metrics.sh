#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

python -u -m grokking_llm measure-stat memo_mia --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap_std --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_counterfactual --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute &
python -u -m grokking_llm measure-stat simplicity_counterfactual --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute &
python -u -m grokking_llm measure-stat loss --config=Z5n7bEDGK4JRT4HyLKSzrw --checkpoint=750,1500,3750,15000,37500 --force-recompute;

wait;
