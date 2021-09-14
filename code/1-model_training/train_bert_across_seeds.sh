#!/bin/bash

DATA_FOLDER=$1
COUNTRY_CODE=$2
MODEL_TYPE_1=$3
TRAIN_TEST_PATH=$4
VENV_PATH=$5
MODEL_PATH=$6

DATA_PATH=${TRAIN_TEST_PATH}/${DATA_FOLDER}/${COUNTRY_CODE}

select_model_name () {
  MODEL_TYPE=$1
  if [ ${MODEL_TYPE} = "dccuchile/bert-base-spanish-wwm-cased" ] || [ ${MODEL_TYPE} = "DeepPavlov/bert-base-cased-conversational" ] || [ ${MODEL_TYPE} = "neuralmind/bert-base-portuguese-cased" ]; then
    MODEL_NAME="bert"
  elif [ ${MODEL_TYPE} = "vinai/bertweet-base" ]; then
    MODEL_NAME="bertweet"
  elif [ ${MODEL_TYPE} = "dlb/electra-base-portuguese-uncased-brwac" ]; then
    MODEL_NAME="electra"
  elif [ ${MODEL_TYPE} = "roberta-base" ] || [ ${MODEL_TYPE} = "mrm8488/RuPERTa-base" ]; then
    MODEL_NAME="roberta"
  elif [ ${MODEL_TYPE} = "xlm-roberta-base" ]; then
    MODEL_NAME="xlmroberta"
  fi
  echo ${MODEL_NAME}
  }

MODEL_NAME_1=$(select_model_name "${MODEL_TYPE_1}")


echo "Launch series of training with different seeds";
for i in {1..15}; do
  sbatch train_bert_model.sbatch ${DATA_FOLDER} ${COUNTRY_CODE} ${MODEL_NAME_1} ${MODEL_TYPE_1} ${i} True ${TRAIN_TEST_PATH} ${VENV_PATH} ${MODEL_PATH}
done

