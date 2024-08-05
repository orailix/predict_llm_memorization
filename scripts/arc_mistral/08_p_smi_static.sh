#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
export OMP_NUM_THREADS=40

# P-SMI dynamic
bash /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/arc_ethics/02_prepare_deploy.sh
python -u -m grokking_llm deploy-cpu --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --only-metrics=p_smi_on_full_dataset --njobs=15;
python -u -m grokking_llm measure-stat p_smi --config=H5gQZhfMsLE5_NTX9hR7TA --checkpoint=50,100,150,200,250,1000,1750,2500 --njobs=34 --force-recompute;
