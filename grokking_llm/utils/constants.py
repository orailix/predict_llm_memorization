# Training config names
TRAIN_CFG_MISTRAL = "mistral"
TRAIN_CFG_LLAMA = "llama"
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
TRAIN_CFG_DEFAULT_LORA_R = 8
TRAIN_CFG_DEFAULT_LORA_ALPHA = 16
TRAIN_CFG_DEFAULT_LORA_DROPOUT = 0.05
TRAIN_CFG_DEFAULT_ACCELERATOR = "cpu"
TRAIN_CFG_DEFAULT_LAST_TOKEN_ONLY = False

# Dataset status
DATASET_BARE_LABEL = "bare"
DATASET_TRUE_LABEL = "true"
DATASET_RANDOM_LABEL = "random"

# Default training arguments
TRAIN_CFG_DEFAULT_TRAINING_ARGS = dict(
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
    evaluation_strategy="steps",
    eval_steps=100,
    do_eval=True,
    remove_unused_columns=False,
    resume_from_checkpoint=True,
    num_train_epochs=1,
)
