#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Prepare deploy
python -u -m grokking_llm deploy-prepare --config=SNFuH5grLjkCFCVDdPB1
