import sys

sys.path.append("/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm")

from grokking_llm.utils import TrainingCfg

# Configs
TrainingCfg.autoconfig(
    "/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/gemma_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/llama_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/mistral_mmlu.json"
).get_output_dir()
