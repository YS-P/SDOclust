#!/bin/bash
set -euo pipefail

PROJ_DIR="${HOME}/SDOclust"
VENV_DIR="${PROJ_DIR}/.venv"
REQ_FILE="${PROJ_DIR}/requirements.txt"
SBATCH_FILE="${PROJ_DIR}/run.sbatch"

mkdir -p "${PROJ_DIR}/logs" "${PROJ_DIR}/results"
cd "${PROJ_DIR}"

echo "1. System deps (python venv)"
sudo apt update -y
sudo apt install -y python3-venv python3-pip

echo "2. Create venv if missing"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

echo "3. Activate venv"
source "${VENV_DIR}/bin/activate"

echo "4. Upgrade pip"
pip install -U pip wheel setuptools

echo "5. Install requirements"
if [ -f "${REQ_FILE}" ]; then
  pip install -r "${REQ_FILE}"
else
  pip install numpy scipy scikit-learn joblib dask distributed
fi

echo "6. Submit Slurm job"
sbatch "${SBATCH_FILE}"

echo "Done. Check queue with: squeue -u \$USER"
