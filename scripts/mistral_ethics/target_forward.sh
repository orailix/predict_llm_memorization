#!/bin/bash

source ~/.bashrc
cd /gpfswork/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -m grokking_llm measure-dyn forward --config=DbFZ_3SZsM2OkESOXcQz2Q
