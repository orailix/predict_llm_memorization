# Dummy CPU test to check deployment scripts are operational

**Deployment id:** `pMODxji4NjwWuG_UmiW4zQ`

**Base config id:** `lvAxlU7wprkJOR9K4h-aAg`

## 1- Init and prepare config

The first scripts (Python) init the deployment config and the base training config. You shoud retrieve the deployment ID and base config ID at the end, and use it in the subsequent scripts.

These scripts can be executed from any location.

```bash
python 01_init_configs.py
bash 02_prepare_deploy.sh
```
