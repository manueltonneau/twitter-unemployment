#!/bin/bash

#SBATCH --job-name=sample_tweets_to_validate_inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_sample_tweets-%j.out

COUNTRY_CODE=$1
MODEL_FOLDER=$2
VENV_PATH=$3
DATA_PATH=$4

module purge
module load anaconda3/2020.02

source ${VENV_PATH}/bin/activate


python3 sample_tweets_to_validate_inference_python.py \
--country_code ${COUNTRY_CODE} \
--model_folder ${MODEL_FOLDER} \
--data_path ${DATA_PATH}