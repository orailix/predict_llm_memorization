# Dummy CPU test to check deployment scripts are operational

**Deployment id:** `SNFuH5grLjkCFCVDdPB1-g`

**Base config id:** `gUY641ZWZR0-s3hU0xSPTA`

## 1- Init and prepare config

The first scripts (Python) init the deployment config and the base training config. You shoud retrieve the deployment ID and base config ID at the end, and use it in the subsequent scripts.

These scripts can be executed from any location.

```bash
python 01_init_configs.py
bash 02_prepare_deploy.sh
```

## 2- Training

- [ ] Verify that all `*.sh` scripts are executable
- [ ] Verify that the paths and names of files are correct will `less` command
