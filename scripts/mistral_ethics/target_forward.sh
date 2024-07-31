#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe
python -u -m grokking_llm measure-dyn forward --config=DbFZ_3SZsM2OkESOXcQz2Q
