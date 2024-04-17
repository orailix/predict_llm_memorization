# Experiment: Mistral Ethics

The experiments must be ran in this order.

## GPU work

#### 1 - GPU: Prepare deployment, train model and compute forward values of the latest checkpoint

```bash
sbatch train_forward_latest.slurm
```

At this point, you should manually check that all model have been successfully trained, for example with:

```bash
upp42qa@jean-zay3:/gpfswork/rech/yfw/upp42qa/grokking_llm/output/individual$ find . -type d -name checkpoint-15600 | wc -l
100
```

#### 2 - GPU: Compute forward values for all checkpoints of the target model

```bash
sbatch target_forward.slurm
```

#### 3 - GPU: Compute forward values on the full dataset for checkpoints 1200 and 15600

```bash
sbatch self_forward.slurm
```

## CPU dynamic measures

#### 3 - CPU: Compute the cpu metrics for the target model (except memorization)

```bash
sbatch cpu_target_metrics.slurm
```

#### 4 - CPU: Compute the memorization metrics of the target model

```bash
sbatch memo_mia_metrics.slurm
```

#### 5 - CPU: Compute the p-smi metric of all models

```bash
sbatch p_smi_full_dataset.slurm
```

## CPU static measures

#### 6 - CPU: Compute the p-smi static metric

```bash
sbatch p_smi_static.slurm
```

#### 7 - CPU: Compute the memorization_membership_inference static metric

```bash
sbatch memo_mia_static.slurm
```

#### 8 - CPU: Compute the counterfactual static metrics

```bash
sbatch counterfactual_static.slurm
```

#### 9 - CPU: Compute the loss static metrics

```bash
sbatch loss.slurm
```

## Prepare local download

#### 10 - CPU: Clean and compress the metrics

**Warning: you should check that the cleaning only affect the compressed forward values. Check script `clean_forward.sh`**

```bash
sbatch clean_forward.slurm
sbatch compress.slurm
```
