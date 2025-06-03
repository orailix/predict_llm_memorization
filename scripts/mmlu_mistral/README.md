# MMLU MISTRAL

**Deployment id:** `Z5n7bEDGK4JRT4HyLKSzrw`

**Base config id:** `Ji5OIUTJGuYCYhAzj6KtZQ`

**Num sample in train:**        30000
**Num sample in split:**        30000 / 2 = 15000
**Batch size | Accumulation:**  2 | 2
**Sample per step:**            2*2 = 4
**Step per epoch:**             15000 / 4 = 3750
**Save steps 1/5 epoch:**       3750 / 5 = 750
**after_c_only_every_n:**       "3750,3750"
**logging_steps:**              250
**num_train_epochs:**           10
