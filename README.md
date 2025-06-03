# Predicting memorization within Large Language Models fine-tuned for classification

Jérémie Dentan<sup>1</sup>, Davide Buscaldi<sup>1, 2</sup>, Aymen Shabou<sup>3</sup>, Sonia Vanier<sup>1</sup>

<sup>1</sup>LIX (École Polytechnique, IP Paris, CNRS) <sup>2</sup>LIPN (Sorbonne Paris Nord) <sup>3</sup>Crédit Agricole SA


## Presentation of the repository

This repository implements the experiments of our preprint "Predicting memorization within Large Language Models fine-tuned for classification".

### Abstract of the paper

Large Language Models have received significant attention due to their abilities to solve a wide range of complex tasks. However these models memorize a significant proportion of their training data, posing a serious threat when disclosed at inference time. To mitigate this unintended memorization, it is crucial to understand what elements are memorized and why. This area of research is largely unexplored, with most existing works providing \textit{a posteriori} explanations. To address this gap, we propose a new approach to detect memorized samples \textit{a priori} in LLMs fine-tuned for classification tasks. This method is effective from the early stages of training and readily adaptable to other classification settings, such as training vision models from scratch. Our method is supported by new theoretical results, and requires a low computational budget. We achieve strong empirical results, paving the way for the systematic identification and protection of vulnerable samples before they are memorized.

### License and Copyright

Copyright 2023-present Laboratoire d'Informatique de Polytechnique. Apache Licence v2.0.

Please cite this work as follows:

```bibtex
@misc{dentan_predicting_2024,
	title = {Predicting memorization within Large Language Models fine-tuned for classification},
	url = {https://arxiv.org/abs/2409.18858},
	author = {Dentan, Jérémie and Buscaldi, Davide and Shabou, Aymen and Vanier, Sonia},
	month = sep,
	year = {2024},
}
```

## Reproducing the CIFAR-10 part of our paper

This repository contains the source code needed to reproduce our results, except the experiments with CIFAR-10 dataset. For these experiments, we provide another repository containing the corresponding source code: [https://github.com/orailix/predict_llm_memorization_cifar10](https://github.com/orailix/predict_llm_memorization_cifar10)

## Overview of the repository

The repository contains three main directories

- `grokking_llm` contains Python source code for the experiments
- `scripts` contains Bash and Slurm scripts for deployment on an HPC cluster
- `figures` contains notebook to reproduce our figures from the paper

**Important notice:** the module we developed is called  `grokking_llm` because the original purpose of this project was to study the Grokking phenomenon on LLM.

### Main configs: `configs` folder

- Appart from the training configs and the deployment configs (see below), two config files are necessary:
  - `main.cfg`: To declare where the HuggingFace cache should be stored (for deploment on an offline HPC cluster, for example), as well as the paths where ouputs and logs should be stored.
  - `env_vars.cfg`: Optionally, to declare environment variables. For example on a HPC cluster with shared CPUS, you might have to use variable `OMP_NUM_THREADS` to make sure that default libraries do not use too many threads compared to what is really available.

### Python source code: `grokking_llm` folder

#### Module `grokking_llm.utils`

- `training_cfg.py` Every training config is mapped to an instance of this class. The instance is associated to an alphanumeric hash (the `config_id`), and all output associated to this training config will be stored in `outputs/individual/<config_id>`. You can use `TrainingCfg.autoconfig` to retrieve any config that was already created.
- `deployment_cfg.py` A deployment config describes the procedure to train models with many training configs. For example, we use deployment config to vary the random split of the dataset between 0 and 99 to train shadow models. Similarly, every deployment config is associated with a `deployment_id` and its outputs stored in `outputs/deployment/<deployment_id>`

#### Module `grokking_llm.training`

- Contains the scripts needed to train models and manage datasets

#### Module `grokking_llm.measures_dyn` and `grokking_llm.measures_stat`

- In appendix A of the paper, we explain the difference between *local* and *global* measures of memorization. In this paper, we use the terms `dynamic` and `static` to refer to these concepts, respectively.
- `grokking_llm.measures_dyn` contains scripts for the *local* measures, i.e. the ones aligned with our threat model: a practitioners willing to audit a *fixed* model trained on a *fixed* dataset.
- `grokking_llm.measures_stat` contains scripts for the *global* measures, i.e. the ones not aligned with our threat model: we obtain *average* vulnerability metrics of a *population* of models trained on random splits of a dataset.

### Figures: `figures` folder

- `01_main_figures.ipynb`: code used for the main figures of the paper
- `01_compare_memorization.ipynb`: code used for figure 6 in the appendix

### Deployment: `scripts` folder

We provide our Bash and Slurm scripts for deplyment on an HPC cluster. We used Jean-Zay HPC cluster from IDRIS. We used some Nvidia A100 80G GPUs and Intel Xeon 6248 CPUs with 40 cores. The training took between 3 and 10 hours on a single GPU. Overall, our experiments are equivalent to around 5000 hours of single GPU and 4000 hours of single-core CPU.

- `arc_mistral`: Deployment scripts for a Mistral 7B model [1] trained on ARC dataset [2].
- `ethics_mistral`: Deployment scripts for a Mistral 7B model [1] trained on ETHICS dataset [3].
- `mmlu_mistral`: Deployment scripts for a Mistral 7B model [1] trained on MMLU dataset [4].
- `mmlu_llama`: Deployment scripts for a Llama 2 7B model [5] trained on MMLU dataset [4].
- `mmlu_gemma`: Deployment scripts for a Gemma 7B model [6] trained on MMLU dataset [4].

## References

- [1] Albert Q. Jiang et al. Mistral 7B, October 2023. http://arxiv.org/abs/2310.06825
- [2] Michael Boratko et al.  Systematic Classification of Knowledge, Reasoning, and Context within the ARC Dataset. In Proceedings of the Workshop on Machine Reading for Question Answering, 2018. http://aclweb.org/anthology/W18-2607
- [3] Dan Hendrycks et al. Aligning AI With Shared Human Values. In ICLR, 2021. https://openreview.net/forum?id=dNy_RKzJacY
- [4] Dan Hendrycks et al. Measuring Massive Multitask Language Understanding. In ICLR, 2021. https://openreview.net/forum?id=d7KBjmI3GmQ
- [5] Hugo Touvron et al. LLaMA: Open and Efficient Foundation
Language Models, February 2023. https://arxiv.org/abs/2302.13971
- [6] Gemma Team et al. Gemma: Open Models Based on Gem-
ini Research and Technology, April 2024. http://arxiv.org/abs/2403.08295

## Acknowledgements

This work received financial support from Crédit Agricole SA through the research chair “Trustworthy and responsible AI” with École Polytechnique. This work was performed using HPC resources from GENCI-IDRIS 2023-AD011014843. We thank Arnaud Grivet Sébert and Mohamed Dhouib for discussions on this paper.
