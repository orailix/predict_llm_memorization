#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

python -u -m grokking_llm measure-stat memo_mia --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap_std --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute &
python -u -m grokking_llm measure-stat memo_counterfactual --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute &
python -u -m grokking_llm measure-stat simplicity_counterfactual --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute &
python -u -m grokking_llm measure-stat loss --config=ZfTT7SCYBAzBtjT3_sy3wg --checkpoint=750,1500,3750,15000,26250,37500 --force-recompute;

wait;
