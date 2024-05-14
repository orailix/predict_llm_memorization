# Training config names
TRAIN_CFG_MISTRAL = "mistral"
TRAIN_CFG_LLAMA = "llama"
TRAIN_CFG_GEMMA = "gemma"
TRAIN_CFG_DUMMY_LLAMA = "dummy_llama"
TRAIN_CFG_ARC = "arc"
TRAIN_CFG_ETHICS = "ethics"
TRAIN_CFG_MMLU = "mmlu"

# Defaults training config values
TRAIN_CFG_DEFAULT_MODEL = TRAIN_CFG_MISTRAL
TRAIN_CFG_DEFAULT_DATASET = TRAIN_CFG_ARC
TRAIN_CFG_DEFAULT_MAX_LEN = 1024
TRAIN_CFG_DEFAULT_LABEL_NOISE = 0.0
TRAIN_CFG_DEFAULT_DATA_SEED = 0
TRAIN_CFG_DEFAULT_SPLIT_ID = 0
TRAIN_CFG_DEFAULT_SPLIT_PROP = 1.0
TRAIN_CFG_DEFAULT_SPLIT_TEST = False
TRAIN_CFG_DEFAULT_LORA_R = 8
TRAIN_CFG_DEFAULT_LORA_ALPHA = 16
TRAIN_CFG_DEFAULT_LORA_DROPOUT = 0.05
TRAIN_CFG_DEFAULT_ACCELERATOR = "cpu"
TRAIN_CFG_DEFAULT_LAST_TOKEN_ONLY = False

# Dataset status
DATASET_BARE_LABEL = -1
DATASET_TRUE_LABEL = 1
DATASET_RANDOM_LABEL = 0

# Maximum number of answers for MCQ
MAX_NUM_MCQ_ANSWER = 16

# Maximum number of sample for MMLU
MMLU_MAX_SIZE = 33750

# Training arguments excluded from config ID
TRAINING_ARGS_EXCLUDED_FROM_CONFIG_ID = [
    "after_c_only_every_n",
    "num_train_epochs",
    "per_device_eval_batch_size",
    "evaluation_strategy",
    "eval_accumulation_steps",
    "num_train_epochs",
    "max_steps",
    "log_level",
    "log_level_replica",
    "log_on_each_node",
    "log_on_each_node",
    "logging_strategy",
    "logging_first_step",
    "logging_steps",
    "logging_nan_inf_filter",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "save_safetensors",
    "save_on_each_node",
    "save_only_model",
    "use_cpu",
    "eval_steps",
    "dataloader_num_workers",
    "run_name",
    "disable_tqdm",
    "report_to",
    "dataloader_persistent_workers",
    "resume_from_checkpoint",
]

# Default training arguments
TRAIN_CFG_DEFAULT_TRAINING_ARGS = dict(
    after_c_only_every_n=None,
    warmup_steps=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=2.5e-5,
    logging_steps=50,
    bf16=True,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=20,
    evaluation_strategy="no",
    remove_unused_columns=False,
    resume_from_checkpoint=True,
    num_train_epochs=1,
)

# DEPLOYMENT
DEPLOYMENT_MAX_LEN = 1e4

# METRIC DEFAULTS
SIGMA_LOGIT_GAP = 8
SMI_N_EST = 2000
SMI_N_NEIGHBORS = 3
MEMO_SCORE_EPSILON = 1e-3
