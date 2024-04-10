#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm deploy-gpu --config=66nIO9d4XxnsfDfc0jaExw --training --forward-latest-on=DbFZ_3SZsM2OkESOXcQz2Q
