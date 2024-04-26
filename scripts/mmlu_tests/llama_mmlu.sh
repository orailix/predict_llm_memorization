#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Training
python -u /gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/setup.py;
python -u -m grokking_llm train --config=686OrtVTBGgc__IZ1Gxzgg;

# Basic measures
python -u -m grokking_llm measure-dyn general --config=686OrtVTBGgc__IZ1Gxzgg ;
python -u -m grokking_llm measure-dyn perf --config=686OrtVTBGgc__IZ1Gxzgg;

# Forward
python -u -m grokking_llm measure-dyn forward --config=686OrtVTBGgc__IZ1Gxzgg;

# Other metrics
python -u -m grokking_llm measure-dyn smi --config=686OrtVTBGgc__IZ1Gxzgg &
python -u -m grokking_llm measure-dyn p_smi --config=686OrtVTBGgc__IZ1Gxzgg &
python -u -m grokking_llm measure-dyn weights --config=686OrtVTBGgc__IZ1Gxzgg &
python -u -m grokking_llm measure-dyn memo_proba_gap --config=686OrtVTBGgc__IZ1Gxzgg &
python -u -m grokking_llm measure-dyn logit_gap --config=686OrtVTBGgc__IZ1Gxzgg &
python -u -m grokking_llm measure-dyn sample_loss --config=686OrtVTBGgc__IZ1Gxzgg;
wait;
