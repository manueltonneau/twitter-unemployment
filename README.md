# Unemployment detection on Twitter

Data and models for ACL 2022 paper "Multilingual Detection of Personal Employment Status on Twitter"

## Data

The labeled tweets can be found in the `data` folder. For each country, the relevant labels can be found in the `<COUNTRY_CODE>.csv` file. In total, the files contain respectively 8376, 11002 and 7156 labeled tweets for the United States (`US`), Mexico (`MX`) and Brazil (`BR`). The tweets are labeled for the 5 classes of interest in the paper, namely whether the tweet indicates that its author "was hired in the past month" (`is_hired_1mo`), "lost her job in the past month" (`lost_job_1mo`), "is looking for a job" (`job_search`), "is unemployed" (`is_unemployed`) or whether the tweet is a job offer (`job_offer`). `1` indicate positive labels while `0` indicate negative labels.

Due to Twitter data sharing policies, we are not able to release the raw tweets and can only release the tweet IDs (`tweet_id`) in combination with the labels. The raw tweets can then be retrieved by using [the Twitter API](https://developer.twitter.com/en/docs/twitter-api/v1/tweets/post-and-engage/api-reference/get-statuses-show-id). 

## Models

We are in the process of uploading the classifiers on the Hugging Face model hub. When this is done, we will update this part with code snippets to download and use our classifiers.

## Citation

We will update this part soon. 



