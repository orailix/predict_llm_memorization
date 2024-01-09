# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil

from grokking_llm.measures import WeightsMetrics
from grokking_llm.training import TrainingCfg, get_model, save_model


def test_weights_metrics():

    # Cleaning
    cfg = TrainingCfg(model="dummy_llama", lora_r=2)
    shutil.rmtree(cfg.get_output_dir())
    cfg.get_output_dir()

    # Saving two checkpoints
    model = get_model(cfg)
    save_model(model, cfg=cfg, at_checkpoint=0)
    save_model(model, cfg=cfg, at_checkpoint=20)

    # Init a metric group
    metrics = WeightsMetrics(cfg)

    # Measures for all checkpoints
    metrics.compute_all_values()

    # Check values
    values = metrics.load_metrics_df()

    # Checks
    assert values.iloc[0]["frob_dist"] == 0
    assert values.iloc[0]["nuc_dist"] == 0
    assert values.iloc[0]["Linf_dist"] == 0
    assert values.iloc[0]["L2_dist"] == 0
    assert values.iloc[1]["frob_dist"] == 0
    assert values.iloc[1]["nuc_dist"] == 0
    assert values.iloc[1]["Linf_dist"] == 0
    assert values.iloc[1]["L2_dist"] == 0
    assert values.iloc[0]["frob_norm"] == values.iloc[1]["frob_norm"]
    assert values.iloc[0]["nuc_norm"] == values.iloc[1]["nuc_norm"]
    assert values.iloc[0]["Linf_norm"] == values.iloc[1]["Linf_norm"]
    assert values.iloc[0]["L2_norm"] == values.iloc[1]["L2_norm"]

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())
