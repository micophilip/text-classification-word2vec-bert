import logging
import os
import os.path

import pandas as pd
from gensim.models import word2vec
from torch.utils.data import Dataset
import random
from collections import deque

PADDING_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def create_vocab_file(directory: str, word2vec_dim: int = 100):
    # Initialize with padding token for shorter sequences and unkown token for out-of-vocabulary tokens.
    vocab = [PADDING_TOKEN + '\n', UNKNOWN_TOKEN + '\n']
    splits = ['train', 'dev']

    if not os.path.isdir(directory):
        raise ValueError(f'{directory} is not a valid directory')
    for split in splits:
        if not os.path.isdir(f'{directory}/{split}'):
            raise ValueError(f'Directory {directory} does not have a {split} sub-folder')

    corpus = []

    for split in splits:
        current_dir = f'{directory}/{split}'
        for filename in os.listdir(current_dir):
            file = os.path.join(current_dir, filename)
            if filename.endswith('.txt') and filename != 'vocab.txt':
                with open(file) as sample:
                    vocab += sample.readlines()
                with open(file) as sample:
                    corpus_lines = [s.replace(' ', '') for s in sample.read().splitlines()]
                    corpus.append(' '.join(corpus_lines) + '\n')

    df = pd.DataFrame({'vocab': vocab})
    unique_vocab = [s.replace(' ', '') for s in df['vocab'].unique()]
    vocab_file = os.path.join(directory, 'vocab.txt')
    corpus_file = os.path.join(directory, 'corpus.txt')
    word2vec_file = os.path.join(directory, 'word2vec')

    with open(vocab_file, 'w+') as output:
        output.writelines(unique_vocab)

    with open(corpus_file, 'w+') as output:
        output.writelines(corpus)

    corpus_vec = word2vec.Word2Vec(corpus_file=corpus_file, vector_size=word2vec_dim, min_count=1, epochs=100)
    corpus_vec.save(word2vec_file)


def split_data(all_data_dir: str):
    split_ratio = [0.6, 0.2, 0.2]
    if sum(split_ratio) != 1.0:
        raise ValueError("split_ratio argument needs to add up to 1")
    if not os.path.isdir(all_data_dir):
        raise ValueError(f"{all_data_dir} is not a valid directory")
    if len(split_ratio) != 3:
        raise ValueError("Please provide ratios for all 3 splits: train, dev, test")

    all_files = os.listdir(all_data_dir)
    all_files_count = len(all_files)
    random.shuffle(all_files)
    files_queue = deque(all_files)
    splits = ['train', 'dev', 'test']
    accumulated_sum = 0

    for i, split in enumerate(splits):
        if i == 2:
            sample_len = all_files_count - accumulated_sum
        else:
            sample_len = int(len(all_files) * split_ratio[i])
            accumulated_sum += sample_len
        os.mkdir(os.path.join(all_data_dir, split))
        for file in range(sample_len):
            filename = files_queue.pop()
            if os.path.isfile(os.path.join(all_data_dir, filename)) and filename.endswith('.txt'):
                os.rename(os.path.join(all_data_dir, filename), os.path.join(all_data_dir, split, filename))
                logger.info(f'Moved {filename} to {split}')


def get_vocab(vocab_file: str):
    if not os.path.isfile(vocab_file):
        raise ValueError(
            f'{vocab_file} does not exist. You need to call python ml/main.py with --create_vocab argument '
            'and pass the directory of the data')

    with open(vocab_file, 'r') as vocab:
        vocabulary = vocab.read().splitlines()

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocab_size = len(vocabulary)

    return word2idx, idx2word, vocab_size


class JSDataset(Dataset):

    def __init__(self, dataset_dir: str, max_seq_len: int, embedding_type: str, word2idx: dict):
        self.samples = []
        self.labels = []
        self.text_lengths = []
        self.sample_ids = []
        self.samples_len = 0
        for filename in os.listdir(dataset_dir):
            file = os.path.join(dataset_dir, filename)
            if filename.endswith('.txt'):
                # Skip wrongly named sample files
                if not filename.endswith('_ATTACK.txt') and not filename.endswith('_BENIGN.txt'):
                    logger.warning(f'Skipping {filename}. It does not end with a _ATTACK.txt or _BENIGN.txt')
                # Find the last index of _ and get the text from that point until before the .txt extension
                # For example: file_name_ATTACK.txt will set label = ATTACK
                label = 1. if filename[filename.rindex('_') + 1:filename.index('.txt')] == 'ATTACK' else 0.
                with open(file, 'r') as sample:
                    sample_point = sample.read().splitlines()
                    sample_point_len = len(sample_point)
                    # Skip empty files
                    if sample_point_len == 0:
                        logger.info(f'Skipping {filename}. It is empty.')
                        continue
                    text_length = sample_point_len
                    target = max_seq_len - 2 if embedding_type == 'bert' else max_seq_len
                    if sample_point_len > target:  # Truncate longer sequences
                        sample_point = sample_point[:target]
                        text_length = max_seq_len
                    elif sample_point_len < target:  # Pad shorter sequences
                        sample_point += [PADDING_TOKEN] * (target - sample_point_len)
                    sample_point = [s.replace(' ', '') for s in sample_point]  # One token per line
                    sample_point = [word if word in word2idx else UNKNOWN_TOKEN for word in sample_point]
                    sample_point = ['[CLS]'] + sample_point + ['[SEP]'] if embedding_type == 'bert' else sample_point
                    self.samples.append(' '.join(sample_point))
                    self.labels.append(label)
                    self.text_lengths.append(text_length)
                    self.sample_ids.append(filename)
        self.samples_len = len(self.samples)
        logger.info(f'Processed dataset from {dataset_dir} containing {self.samples_len} samples.')

    def __len__(self):
        return self.samples_len

    def __getitem__(self, idx):
        return {'text': self.samples[idx], 'label': self.labels[idx], 'text_length': self.text_lengths[idx], 'sample_id': self.sample_ids[idx]}
