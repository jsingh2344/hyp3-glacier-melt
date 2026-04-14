#!/bin/bash --login
set -e
conda activate hyp3-glacier-melt
exec python -um hyp3_glacier_melt "$@"
