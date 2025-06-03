#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=40

# P-SMI dynamic
bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/ethics_mistral/02_prepare_deploy.sh
python -u -m grokking_llm deploy-cpu --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --only-metrics=p_smi_on_full_dataset --njobs=15;
python -u -m grokking_llm measure-stat p_smi --config=JNeyT5OUP3GFtJ_QNojppg --checkpoint=397,794,1191,1588,1985,7940,13895,19850 --njobs=34 --force-recompute;
