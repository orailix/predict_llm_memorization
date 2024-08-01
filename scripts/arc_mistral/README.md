# MMLU MISTRAL

**Deployment id:** `TODO`

**Base config id:** `TODO`

**Num sample in train:**        2000
**Num sample in split:**        2036 // 2 = 1000
**Batch size | Accumulation:**  2 | 2
**Sample per step:**            2*2 = 4
**Step per epoch:**             1018 // 4 = 250
**Save steps 1/5 epoch:**       255 / 5 = 50
**after_c_only_every_n:**       "250,250"
**logging_steps:**              50
**num_train_epochs:**           10
