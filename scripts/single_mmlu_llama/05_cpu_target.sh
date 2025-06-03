#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=20;

# CPU computations
python -u -m grokking_llm measure-dyn general --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn perf --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn smi --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn p_smi --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn weights --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn logit_gap --config=Fp-lMrMPD6Br3DT_wCfPMA &
python -u -m grokking_llm measure-dyn sample_loss --config=Fp-lMrMPD6Br3DT_wCfPMA &

wait;
