#!/bin/bash
#SBATCH --job-name=sdoclust_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28GB

set -euo pipefail

source "${HOME}/SDOclust/.venv/bin/activate"

mkdir -p logs results

# Set Experiment
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

    srun --cpu-bind=cores --cpus-per-task="${CORES}" \
      python sdoclust_parallel.py \
        --datasets "blobs,noisy_blobs" \
        --size "${SIZE}" \
        --backends "seq,joblib,dask" \
        --splits "2,4,8,16" \
        --noise "0.05,0.15" \
        --centers "5,10" \
        --seeds "42" \
        --output "results/results_core_${CORES}.csv"
  done
done

echo "All experiments completed. Generating plots."
python plot_results.py
echo "Pipeline Finished"