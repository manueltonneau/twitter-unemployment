#!/bin/bash

#SBATCH --job-name=adaptive_retrieval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_adaptive_retrieval-%j.out

COUNTRY_CODE=$1
INFERENCE_FOLDER=$2
ITERATION_NUMBER=$3
VENV_PATH=$4
DATA_PATH=$5

module purge
module load anaconda3/2020.02

source ${VENV_PATH}/bin/activate


python3 retrieve_top_tweets.py \
--country_code ${COUNTRY_CODE} \
--inference_folder ${INFERENCE_FOLDER} \
--iteration_number "${ITERATION_NUMBER}" \
--data_path ${DATA_PATH}