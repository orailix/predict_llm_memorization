# Experiment: Mistral Ethics

The experiments must be ran in this order.

## 1 - GPU: Prepare deployment, train model and compute forward values of the latest checkpoint

```bash
bash _prepare_deploy.sh
sbatch train_forward_latest.slurm
```

At this point, you should manually check that all model have been successfully trained, for example with:

```bash
upp42qa@jean-zay3:/gpfswork/rech/yfw/upp42qa/grokking_llm/output/individual$ find . -type d -name checkpoint-15600 | wc -l
100
```

## 2 - GPU: Compute forward values for all checkpoints of the target model

```bash
sbatch target_forward.slurm
```

## 3 - GPU: Compute forward values on the full dataset for checkpoints 1200 and 15600

```bash
bash _prepare_deploy.sh
sbatch self_forward.slurm
```

## 3 - CPU: Compute the cpu metrics for the target model (except memorization)

```bash
sbatch cpu_target_metrics.slurm
```

## 4 - CPU: Compute the memorization metrics of the target model

```bash
sbatch memo_mia_metrics.slurm
```

## 5 - CPU: Clean and compress the metrics

**Warning: you should check that the cleaning only affect the compressed forward values. Check script `clean_forward.sh`**

```bash
sbatch clean_forward.slurm
bash _prepare_deploy.sh
sbatch compress.slurm
```
