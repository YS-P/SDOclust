#!/bin/bash
#SBATCH --job-name=sdoclust_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

cd ~/SDOclust
mkdir -p logs results

# Activate venv
source ~/SDOclust/.venv/bin/activate

# Set experiment
CORES_LIST=(1 2 4 8 16)
SIZE_LIST=(50000 200000)

# Prevent BLAS/OMP corruption
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPU_BIND=cores

for CORES in "${CORES_LIST[@]}"; do
  for SIZE in "${SIZE_LIST[@]}"; do
    echo "Running SIZE=${SIZE}, CORES=${CORES}"

    srun --ntasks=1 --cpu-bind=cores --cpus-per-task="${CORES}" \
      python main.py \
        --datasets "blobs,noisy_blobs" \
        --size "${SIZE}" \
        --backends "seq,joblib,dask" \
        --splits "2,4,8,16" \
        --seeds "42"
  done
done
