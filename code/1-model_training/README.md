# Fine-tuning BERT-based models

## Preliminary steps:

- Ideally, create a virtual environment specifically for training and activate it:

```
$ python3 -m virtualenv env_name
$ source env_name/bin/activate
```

- Install the necessary packages:
```
$ cd ./twitter-unemployment/code/X-model_training
$ pip install -r requirements.txt
```

- Install PyTorch separately without cache to not use too much memory:
`$ pip install --no-cache-dir torch==1.5.0`

- Install [apex](https://github.com/nvidia/apex) to be able to use fp16 training. On Linux, it is done this way:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

In case the first version causes errors, another possible solution to install apex is to replace the final line by:

`$ pip install -v --no-cache-dir ./`

We are now ready to start fine-tuning.

## Training command:

We used a SLURM cluster to fine-tune our BERT-based models. If you train on other types of clusters, you just need to remove the SLURM-specific parameters at the top of the `train_bert_model.sbatch`.
To train a binary BERT-based classifier on all 5 classes on the cluster, run:

`$ sbatch train_bert_model.sbatch <DATA_FOLDER_NAME> <MODEL_NAME> <MODEL_TYPE> <INTRA_EPOCH_EVALUATION> <TRAIN_TEST_PATH> <VENV_PATH> <MODEL_PATH>`

with:
- <DATA_FOLDER_NAME>: the name of the folder where the train/val CSVs are stored. We included the date of the folder creation and the active learning iteration number in the folder name for clarity (e.g. `jul23_iter0`).
- <MODEL_NAME>: the BERT-based model architecture used. By default, it is always set to `bert`. 
- <MODEL_TYPE>: the type of BERT-based model used (e.g. `DeepPavlov/bert-base-cased-conversational` for Conversational BERT). This refers to the shortcut name of the model on the HuggingFace hub. The whole list can be found [here](https://huggingface.co/transformers/pretrained_models.html). 
- <INTRA_EPOCH_EVALUATION>: a string parsed as a boolean to determine whether to perform intra-epoch evaluation (10 per epoch by default). Possible values are `t` (parsed as `True`) or `f` (parsed as `False`).
- <TRAIN_TEST_PATH>: the path to the folder `<DATA_FOLDER_NAME>`.
- <VENV_PATH>: the path to the virtual environment.
- <MODEL_PATH>: the path where the fine-tuned models will be stored.


Example command: 

`$ sbatch train_bert_model.sbatch jul23_iter0 bert DeepPavlov/bert-base-cased-conversational t my/train/test/path my/venv/path my/model/path`

## Results:

### Models:

The trained models are automatically saved at `<MODEL_PATH>`.

A version of the model is saved for each epoch i at `${MODEL_TYPE}_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}/${class}/models/checkpoint-${NB_STEPS}-epoch-${i}` where `class` is in `['lost_job_1mo', 'is_hired_1mo', 'job_search', 'job_offer', 'is_unemployed']`. 

The best model in terms of evaluation loss is saved in the folder: `${MODEL_TYPE}_${DATA_FOLDER_NAME}_${SLURM_JOB_ID}/${class}/models/best_model`. 

### Performance metrics and scores:

For each class and training round, four CSVs are saved:
- one with performance metrics (Precision/Recall/F1/AUC) on evaluation set, including metadata such as job_id, date and time, path of data used. It is called `val_${class}_evaluation.csv`. 
- one with performance metrics (//) on holdout set ( // ). It is called `holdout_${class}_evaluation.csv`. 
- one with scores for the evaluation set. It is called `val_${class}_scores.csv`. 
- one with scores for the holdout set. It is called `holdout_${class}_scores.csv`. 

Note that Precision/Recall/F1 are computed for a threshold of 0.5.

These four CSVs are saved at: `<TRAIN_TEST_PATH>/<DATA_FOLDER_NAME>/results/<MODEL_TYPE>_<SLURM_JOB_IB>`. 
