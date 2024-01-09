# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil

from grokking_llm.training import (
    TrainingCfg,
    add_labels,
    format_dataset,
    get_dataset,
    get_model,
    get_trainer,
    save_model,
    tokenize_dataset,
)


def test_get_trainer():

    cfg = TrainingCfg(model="dummy_llama")

    # Dataset - test
    test_dataset = get_dataset(cfg, split="test")
    test_dataset_formatted = format_dataset(test_dataset, cfg)
    test_dataset_labelled = add_labels(test_dataset_formatted, cfg, "test")
    test_dataset_tokenized = tokenize_dataset(test_dataset_labelled, cfg)

    # Model
    model = get_model(cfg)

    # Trainer
    trainer = get_trainer(
        cfg,
        model=model,
        train_dataset=test_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
    )

    # Checks
    assert trainer.last_token_only == cfg.last_token_only
    for key, value in cfg.training_args.items():
        assert key in dir(trainer.args)
        assert getattr(trainer.args, key) == value

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())


def test_trainer_resume_from_checkpoint():

    cfg = TrainingCfg(
        model="dummy_llama",
        accelerator="cpu",
        training_args=dict(resume_from_checkpoint=True, max_steps=1),
    )

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())
    cfg.get_output_dir()

    # Model
    model = get_model(cfg)

    # Dataset - test
    test_dataset = get_dataset(cfg, split="test")
    test_dataset_formatted = format_dataset(test_dataset, cfg)
    test_dataset_labelled = add_labels(test_dataset_formatted, cfg, "test")
    test_dataset_tokenized = tokenize_dataset(test_dataset_labelled, cfg)

    # Init trainer -- It should create a model from scratch
    trainer = get_trainer(
        cfg,
        model=model,
        train_dataset=test_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
    )

    # The model should be saved on the disk
    assert 0 in cfg.get_available_checkpoints()
    model_reloaded = get_model(cfg, at_checkpoint=0)
    for param_before, param_reloaded in zip(
        trainer.model.parameters(),
        model_reloaded.parameters(),
    ):
        assert (param_before == param_reloaded).all()

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())
