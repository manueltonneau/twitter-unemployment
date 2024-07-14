# Unemployment detection on Twitter

Data and models for ACL 2022 paper ["Multilingual Detection of Personal Employment Status on Twitter"](https://aclanthology.org/2022.acl-long.453/)

## Data

An anonymized version of the dataset can be found on :hugs: under [this link](https://huggingface.co/datasets/worldbank/twitter-labor-market).

## Models

Our model have been open-sourced on :hugs: and can be found in [this collection](https://huggingface.co/collections/worldbank/twitter-labor-market-insights-66939781e32997bdf7663e1f). Please find in the table below the model names in the hub for each language and class.

| Class                            | English  | Spanish | Portuguese
| -------------------------------- | ---------| --------| --------------------------------------------------------------------------------------------
|  Is Unemployed | `worldbank/bert-twitter-en-is-unemployed` | `worldbank/bert-twitter-es-is-unemployed` | `worldbank/bert-twitter-pt-is-unemployed`
| Lost Job | `worldbank/bert-twitter-en-lost-job` | `worldbank/bert-twitter-es-lost-job` | `worldbank/bert-twitter-pt-lost-job` 
| Job Search | `worldbank/bert-twitter-en-job-search` | `worldbank/bert-twitter-es-job-search` |  `worldbank/bert-twitter-pt-job-search` 
| Is Hired | `worldbank/bert-twitter-en-is-hired` | `worldbank/bert-twitter-es-is-hired` | `worldbank/bert-twitter-pt-is-hired` 
| Job Offer | `worldbank/bert-twitter-en-job-offer` | `worldbank/bert-twitter-es-job-offer` | `worldbank/bert-twitter-pt-job-offer` 

To use a specific model, say the English model for the class Is Unemployed, do the following:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("worldbank/bert-twitter-en-is-unemployed")
model = AutoModel.from_pretrained("worldbank/bert-twitter-en-is-unemployed")
```
## Citation

If you find our work useful, please cite:

```
@inproceedings{tonneau-etal-2022-multilingual,
    title = "Multilingual Detection of Personal Employment Status on {T}witter",
    author = "Tonneau, Manuel  and
      Adjodah, Dhaval  and
      Palotti, Joao  and
      Grinberg, Nir  and
      Fraiberger, Samuel",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.453",
    doi = "10.18653/v1/2022.acl-long.453",
    pages = "6564--6587",
}
```


