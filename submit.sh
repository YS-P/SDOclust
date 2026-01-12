#!/bin/bash
set -euo pipefail

mkdir -p logs results
sbatch run_slurm.sbatch
