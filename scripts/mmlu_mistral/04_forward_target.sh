#!/bin/bash

source ~/.bashrc
cd /lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm
conda activate expe

# Measure forward
python -u -m grokking_llm measure-dyn forward --config=Ji5OIUTJGuYCYhAzj6KtZQ;
