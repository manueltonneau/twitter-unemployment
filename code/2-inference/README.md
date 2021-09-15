# Launching inference with BERT-based models on 100M samples

Here, we detail how we run inference with our BERT-based classifiers on large random samples, after we have selected the best performing model for each class.

## Preliminary steps:

Before we can start, we need to report the folder names corresponding to the best models. To do so, for each class `A`, iteration `i` and country `C`, you need to:
- retrieve the best seeds. These are outputted by the `evaluation_across_seeds.py` script, as described at the end of the README from the `1-model_training` folder.
- find the related folder corresponding to the model trained on this best seed.
- copy the folder name containing the models in the `best_model_folders_dict` dictionary. The folder name should be the value of `best_model_folders_dict['C']['iteri']['A']`. If the method you use is exploit-explore retrieval, adaptive retrieval, uncertainty sampling on calibrated scores or uncertainty sampling on uncalibrated scores, you should modify modify the value of the dict for the `method` argument being respectively 0, 1, 2 or 3.

## Launch inference:

### 1. Convert PyTorch models to ONNX format:

We use ONNX to speed up the inference of the BERT-based models. To convert a PyTorch model to ONNX, run: 
```
$ sbatch onnx_model_conversion.sbatch <COUNTRY_CODE> <ITER> <METHOD> <VENV_PATH> <MODEL_PATH>
```
where:
- `<COUNTRY_CODE>` is in `['US', 'MX', 'BR']`
- `<ITER>` refers to the active learning iteration number (starts at 0)
- `<METHOD` refers to the active learning method used (`0` for exploit explore retrieval, `1` for adaptive retrieval, `2` for uncertainty sampling on calibrated scores, `3` on uncertainty sampling with uncalibrated scores).
- `<VENV_PATH>`: the path to the virtual environment.
- `<MODEL_PATH>`: the path where the fine-tuned models will be stored.

When running the above command, the best models for each class for the specified iteration and country, which are listed in the dictionary `best_model_folders_dict` in the script `onnx_model_conversion.py`, are converted to ONNX, optimized and quantized.

### 2. Run inference

Once the models are converted to ONNX, they can be used to run inference on the random samples. To do so, run:

```
$ sbatch --array=0-1999 inference_ONNX_bert_100M_random.sbatch <COUNTRY_CODE> <ITER> <MODE> <RUN_NAME> <METHOD> <VENV_PATH> <RANDOM_SAMPLE_PATH> <MODEL_PATH> <OUTPUT_PATH>
```

where: 
- `<COUNTRY_CODE>` and `<ITER>` are defined above
- `<MODE>`: there are two 10M random samples. One, which we'll call the "evaluation random sample" is only intended to evaluate the model's performance. The other one, which we'll call the "active learning random sample" is used to pick up and label new tweets through active learning.
  - `<MODE>` is equal to 0 if the inference is to be run on the evaluation random sample
    
  - `<MODE>` is equal to 1 if the inference is to be run on the active learning random sample
    
- `<RUN_NAME`: a name to differentiate output folders. The output folder name is defined as:
    - `iter_${ITER}-${RUN_NAME}-${SLURM_JOB_ID}-evaluation` if `<MODE> = 0`
    - `iter_${ITER}-${RUN_NAME}-${SLURM_JOB_ID}-new_samples` if `<MODE> = 1`
- `<METHOD` refers to the active learning method used (`0` for exploit explore retrieval, `1` for adaptive retrieval, `2` for uncertainty sampling on calibrated scores, `3` on uncertainty sampling with uncalibrated scores).
- `<VENV_PATH>`: the path to the virtual environment.
- `<RANDOM_SAMPLE_PATH>`: the path to the folder where the random samples are stored.
- `<MODEL_PATH>`: the path where the fine-tuned models will be stored.
- `<OUTPUT_PATH>`: the path to the folder where the inferences will be stored.

## Results:

### Inference files

For each label `<LABEL>`, all tweets with their respective score are saved at `<OUTPUT_PATH>/output/<LABEL>`. 

The logs are saved at `<OUTPUT_PATH>/logs/<LABEL>`.