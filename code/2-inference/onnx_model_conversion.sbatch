#!/bin/bash

#SBATCH --job-name=ONNX_conversion
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_ONNX_conversion_%j.out

COUNTRY_CODE=$1
ITER=$2
METHOD=$3
VENV_PATH=$4
MODEL_PATH=$5

module purge
module load anaconda3/2020.02

source ${VENV_PATH}/bin/activate

echo "pyenv activated"

python3 onnx_model_conversion.py \
--iteration_number ${ITER} \
--country_code ${COUNTRY_CODE} \
--method ${METHOD} \
--model_path ${MODEL_PATH}

exit