from grokking_llm.utils import TrainingCfg

TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/gemma_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/llama_mmlu.json"
).get_output_dir()
TrainingCfg.autoconfig(
    "/gpfswork/rech/yfw/upp42qa/grokking_llm/scripts/mmlu_tests/mistral_mmlu.json"
).get_output_dir()
