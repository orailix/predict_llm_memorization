#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

python -u -m grokking_llm measure-stat memo_mia --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute &
python -u -m grokking_llm measure-stat memo_logit_gap_std --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute &
python -u -m grokking_llm measure-stat memo_counterfactual --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute &
python -u -m grokking_llm measure-stat simplicity_counterfactual --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute &
python -u -m grokking_llm measure-stat loss --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --force-recompute;

wait;
