import pandas as pd
import argparse
import os
from collections import defaultdict
import logging
from pathlib import Path
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--data_folder", type=str,
                        default="jan5_iter0")
    parser.add_argument("--train_test_path", type=str)
    args = parser.parse_args()
    return args


def build_auc_dict(results_dict: dict, model_type: str) -> dict:
    """
    Compute AUC statistics for each label and across models.
    Adapted from https://stackoverflow.com/questions/36053321/python-average-all-key-values-with-same-key-to-dictionary
    """
    by_label = defaultdict(list)
    for model, mrs in results_dict[model_type].items():
        for label_key, auc_value in mrs.items():
            by_label[label_key].append(auc_value)

    return {label_key: {'mean': sum(auc_value_list) / len(auc_value_list),
                        # 'std': statistics.stdev(auc_value_list),
                        'min': min(auc_value_list),
                        'max': max(auc_value_list)
                        } for label_key, auc_value_list in by_label.items()}


if __name__ == '__main__':
    args = get_args_from_command_line()
    output_path = f'{args.train_test_path}/{args.country_code}/{args.data_folder}/evaluation'
    results_folder = f'{args.train_test_path}/{args.country_code}/{args.data_folder}/train_test/results'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    results_folders_list = os.listdir(results_folder)
    results_dict = dict()

    for model in ['neuralmind-bert-base-portuguese-cased', 'DeepPavlov-bert-base-cased-conversational',
                  'dccuchile-bert-base-spanish-wwm-cased']:
        for seed in range(1, 16):
            r = re.compile(f'{model}_[0-9]+_seed-{str(seed)}$')
            folder_name_str = list(filter(r.match, results_folders_list))
            if len(folder_name_str) > 0:
                folder_name_str = list(filter(r.match, results_folders_list))[0]
                for label in ['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']:
                    if label not in results_dict.keys():
                        results_dict[label] = dict()
                    path_data = os.path.join(results_folder, folder_name_str, f'val_{label}_evaluation.csv')
                    if Path(path_data).exists():
                        df = pd.read_csv(
                            os.path.join(results_folder, folder_name_str, f'val_{label}_evaluation.csv'),
                            index_col=0)
                        results_dict[label][f'auc_{model}_{str(seed)}'] = float(df['value']['auc'])
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.round(3)
    results_df = results_df.reset_index()
    results_df['model'] = results_df['index'].apply(lambda x: x.split('_')[1])
    results_df['seed'] = results_df['index'].apply(lambda x: x.split('_')[2])
    results_df = results_df[
        ['model', 'seed', 'lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']]
    results_df.to_csv(os.path.join(output_path, 'auc_results.csv'))
    results_df = results_df.set_index(['seed'])
    results_df = results_df[['lost_job_1mo', 'is_hired_1mo', 'is_unemployed', 'job_offer', 'job_search']]
    print(results_df.idxmax())
    print(results_df.max())
