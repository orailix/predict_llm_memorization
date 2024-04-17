#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm measure-dyn general --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn perf --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn smi --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn p_smi --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn weights --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn memo_proba_gap --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn memo_logit_gap --config=DbFZ_3SZsM2OkESOXcQz2Q
python -u -m grokking_llm measure-dyn sample_loss --config=DbFZ_3SZsM2OkESOXcQz2Q
