import pandas as pd
from glob import glob
import os
import logging
import statistics

pd.set_option('display.max_rows', None)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

BENIGN_LABEL = 'BENIGN'
MALICIOUS_LABEL = 'ATTACK'


def save_predictions(experiment_dir: str, metrics: dict):
    predictions = {k: metrics[k] for k in ['sample_id', 'y', 'y_hat', 'confidence']}
    predictions_df = pd.DataFrame.from_dict(predictions)
    predictions_df.to_csv(experiment_dir, index=False)


def get_class_count(split_dir: str, label: str) -> int:
    return len([y for x in os.walk(split_dir) for y in glob(os.path.join(x[0], f'*_{label}.txt'))])


def dataset_analytics(data_dir: str):
    splits = ['train', 'dev', 'test']
    all_benign_count = 0
    all_malicious_count = 0
    dataset_name = data_dir[data_dir.rindex("/") + 1:]
    all_sequence_lengths = []
    all_tokens = []
    all_class = []
    all_splits = []

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        benign_len = get_class_count(split_dir, BENIGN_LABEL)
        all_benign_count += benign_len
        malicious_len = get_class_count(split_dir, MALICIOUS_LABEL)
        all_malicious_count += malicious_len
        split_count = benign_len + malicious_len
        assert split_count != 0, f'Dataset {dataset_name} {split} split does not contain any files'
        malicious_ratio = int((malicious_len / split_count) * 100)
        logger.info(f'Total {split} split is {split_count}')
        logger.info(f'Dataset {dataset_name} {split} split contains {malicious_ratio}:{100 - malicious_ratio} malicious:benign split')
        sequence_lengths = []
        for filename in os.listdir(split_dir):
            label = BENIGN_LABEL if filename.endswith(f'_{BENIGN_LABEL}.txt') else MALICIOUS_LABEL
            file = os.path.join(split_dir, filename)
            with open(file) as sample:
                tokens = sample.read().splitlines()
                tokens = ['fn:' if s.startswith('fn:') else s for s in tokens]
                seq_len = len(tokens)
                sequence_lengths.append(seq_len)
                all_sequence_lengths.extend(sequence_lengths)
                all_tokens.extend(tokens)
                all_class.extend([label] * seq_len)
                all_splits.extend([split] * seq_len)
        logger.info(f'Dataset {dataset_name} {split} split contains {statistics.mean(sequence_lengths)} average sequence length')

    dataset_count = all_benign_count + all_malicious_count
    all_malicious_ratio = int((all_malicious_count / dataset_count) * 100)

    logger.info(f'Total data files in {dataset_name} split is {dataset_count}')
    logger.info(f'Dataset {dataset_name} contains {all_malicious_ratio}:{100 - all_malicious_ratio} malicious:benign split')
    logger.info(f'Average sequence length for {dataset_name} is {statistics.mean(all_sequence_lengths)}')

    df = pd.DataFrame.from_dict({'token': all_tokens, 'class': all_class, 'split': all_splits, 'count': [1] * len(all_tokens)})
    df = df.groupby(['token', 'class'], as_index=False).sum().sort_values(by=['token', 'class', 'count'])
    df.to_csv(f'{data_dir}/summary_stats.csv')
    logger.info(f'Feature count by class \n{df}')
