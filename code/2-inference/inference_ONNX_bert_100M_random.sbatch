#!/bin/bash

#SBATCH --job-name=ONNX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_inference_berts_%j.out

COUNTRY_CODE=$1
ITER=$2
MODE=$3
RUN_NAME=$4
METHOD=$5
VENV_PATH=$6
RANDOM_SAMPLE_PATH=$7
MODEL_PATH=$8
OUTPUT_PATH=$9

JOB_ID=${SLURM_ARRAY_JOB_ID}


if [ ${MODE} -eq 0 ]
then
  INPUT_PATH=${RANDOM_SAMPLE_PATH}/${COUNTRY_CODE}/evaluation
  OUTPUT_PATH=${OUTPUT_PATH}/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-evaluation
elif [ ${MODE} -eq 1 ]
then
  INPUT_PATH=${RANDOM_SAMPLE_PATH}/${COUNTRY_CODE}/new_samples
  OUTPUT_PATH=${OUTPUT_PATH}/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-new_samples
elif [ ${MODE} -eq 2 ]
then
  INPUT_PATH=${RANDOM_SAMPLE_PATH}/${COUNTRY_CODE}/test
  OUTPUT_PATH=${OUTPUT_PATH}/${COUNTRY_CODE}/iter_${ITER}-${RUN_NAME}-${JOB_ID}-test
fi

OUTPUT_MODELS=${OUTPUT_PATH}/output
OUTPUT_LOGS=${OUTPUT_PATH}/logs

# Create output folders if they don't exist
mkdir -p ${OUTPUT_MODELS}
mkdir -p ${OUTPUT_LOGS}


module purge
module load anaconda3/2020.02

source ${VENV_PATH}/bin/activate

echo "pyenv activated"

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
      echo SLURM_ARRAY_TASK_ID=test
fi
echo 'running inference..'
srun time python -u inference_ONNX_bert_100M_random.py --input_path ${INPUT_PATH} --output_path ${OUTPUT_MODELS} --country_code ${COUNTRY_CODE} --iteration_number ${ITER} --method ${METHOD} --model_path ${MODEL_PATH} > ${OUTPUT_LOGS}/${SLURM_ARRAY_TASK_ID}-$(date +%s).log 2>&1
echo 'done'

exit



