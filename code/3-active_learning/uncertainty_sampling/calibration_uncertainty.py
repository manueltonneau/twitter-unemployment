import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from config import *
import pickle
import warnings
import os
import re

warnings.filterwarnings('ignore')

# number of times to sample data
num_samples = 10000
print(f'Calibrating with {num_samples}')

path_data = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/US'
fig_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/code/2-twitter_labor/3-model_evaluation/expansion/preliminary'
params_dict = {}

iter_names_list = ['iter_3-convbert_uncertainty-6318280-evaluation']

iter_number = int(re.findall('iter_(\d)', iter_names_list[0])[0])
for label in ['is_hired_1mo', 'lost_job_1mo', 'is_unemployed', 'job_search', 'job_offer']:
    params_dict[label] = {}
    for i, iter_name in enumerate(iter_names_list):
        # load data
        df = pd.read_csv(f'{path_data}/{iter_name}/{label}.csv')
        params = []
        params_dict[label][iter_name] = {}
        # get the positives
        positives_df = df[df['class'] == 1]
        # negatives
        negatives_df = df[df['class'] == 0]
        # sample min(len(positives_df), len(negatives_df)) rows from the data without replacement
        for negative_df in [negatives_df.sample(n=min(len(positives_df), len(negatives_df)), replace=False) for _ in
                            range(num_samples)]:
            dp = positives_df.sample(n=len(positives_df), replace=True)
            dn = negative_df.sample(n=len(negative_df), replace=True)
            d = pd.concat([dp, dn])
            # build logistic regression model to fit data
            # get the scores and labels
            X = np.asarray(d['score']).reshape((-1, 1))
            y = np.asarray(d['class'])
            # perform calibration using sigmoid function with 5 cv
            try:
                model = LogisticRegression(penalty='none').fit(X, y)
            except:
                print(f'failed with {iter_name} on label {label}')
                continue
            # get all A, B for each of the model
            params.append([model.coef_[0][0], model.intercept_[0]])

        print(f'Sampled {len(positives_df)} positives for {label}, {iter_name}')
        # calculate the calibrated score:  avg(logit(ax+b))
        all_calibrated_scores = [1 / (1 + np.exp(-(param[0] * df['score'] + param[1]))) for param in params]
        df['Calibrated score'] = np.mean(all_calibrated_scores, axis=0)
        params_dict[label][iter_name]['params'] = params

pickle.dump(params_dict, open(f'{fig_path}/calibration_dicts/calibration_dict_uncertainty_{num_samples}_iter{iter_number}.pkl', 'wb'))
