#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Training
python -u -m grokking_llm train --config=lvAxlU7wprkJOR9K4h-aAg;

# Basic measures
python -u -m grokking_llm measure-dyn general --config=lvAxlU7wprkJOR9K4h-aAg ;
python -u -m grokking_llm measure-dyn perf --config=lvAxlU7wprkJOR9K4h-aAg;

# Forward
python -u -m grokking_llm measure-dyn forward --config=lvAxlU7wprkJOR9K4h-aAg;

# Other metrics
python -u -m grokking_llm measure-dyn smi --config=lvAxlU7wprkJOR9K4h-aAg &
python -u -m grokking_llm measure-dyn p_smi --config=lvAxlU7wprkJOR9K4h-aAg &
python -u -m grokking_llm measure-dyn weights --config=lvAxlU7wprkJOR9K4h-aAg &
python -u -m grokking_llm measure-dyn memo_proba_gap --config=lvAxlU7wprkJOR9K4h-aAg &
python -u -m grokking_llm measure-dyn memo_logit_gap --config=lvAxlU7wprkJOR9K4h-aAg &
python -u -m grokking_llm measure-dyn sample_loss --config=lvAxlU7wprkJOR9K4h-aAg;
wait;
