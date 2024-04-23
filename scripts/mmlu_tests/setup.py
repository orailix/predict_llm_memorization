from grokking_llm.training import get_dataset, get_model
from grokking_llm.utils import TrainingCfg

# Configs
TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/gemma_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/llama_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/mistral_mmlu.json"
).get_output_dir()

# Models
get_model(TrainingCfg(model="gemma"))
get_model(TrainingCfg(model="llama"))
get_model(TrainingCfg(model="mistral"))

# Dataset
get_dataset(TrainingCfg(dataset="mmlu"))
