#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

python -u -m grokking_llm measure-stat memo_mia --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap_std --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute &
python -u -m grokking_llm measure-stat memo_counterfactual --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute &
python -u -m grokking_llm measure-stat simplicity_counterfactual --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute &
python -u -m grokking_llm measure-stat loss --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --force-recompute;

wait;
