#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-data/robomimic}

python -m robomimic.scripts.download_datasets \
    --download_dir "$DATA_DIR" \
    --tasks lift can square transport tool_hang \
    --dataset_types ph \
    --hdf5_types low_dim

echo "Done. Datasets under $DATA_DIR/<task>/ph/low_dim_v15.hdf5"
