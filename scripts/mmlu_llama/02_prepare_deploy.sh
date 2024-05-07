#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Prepare deploy
python -u -m grokking_llm deploy-prepare --config=ZfTT7SCYBAzBtjT3_sy3wg
