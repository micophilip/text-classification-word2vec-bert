import argparse
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from pytz import timezone
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_vocab_file, JSDataset, get_vocab, split_data
from evaluate import evaluate
from model import JSMalCatcherModel
from embeddings import BERTEmbedding, Word2VecEmbedding
from train import train
from analytics import save_predictions, dataset_analytics
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import statistics

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='JS Mal Catcher Modeling')

parser.add_argument('--create_vocab', type=str, required=False,
                    help="When passed, expects a directory of text files to create vocabulary. It creates a vocab.txt file in the same directory")
parser.add_argument('--data_dir', type=str, required=False, help="Directory containing train and test subfolders")
parser.add_argument('--model_dir', type=str, required=False, help="Directory containing train and test subfolders")
parser.add_argument('--train', action='store_true', help="Train flag")
parser.add_argument('--test', type=str, required=False, action='store', nargs='?',
                    help="When passed without a train flag, it expects the model version. When passed with --train it tests the model produced by training")
parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size for training')
parser.add_argument('--n_epochs', type=int, default=5, required=False, help="Number of epochs to train")
parser.add_argument('--max_seq_len', type=int, default=128, required=False,
                    help="Vectors need to be of the same size. This argument specifies when truncating for longer sequence or padding for shorter sequence"
                         " happens")
parser.add_argument('--lr', type=float, default=1e-3, required=False, help="Learning rate for the optimizer")
parser.add_argument('--embedding_type', type=str, default='default', required=False, choices=['default', 'bert', 'word2vec'], help="Embedding type to use")
parser.add_argument('--embedding_dim', type=int, default=100, required=False, help="Input embedding dimension")
parser.add_argument('--split_data', required=False, action="store_true", help="Randomly split .txt data in the provided folder to train, dev and test splits")
parser.add_argument('--wilcoxon_pairs', type=str, required=False, help='Pairs of experiments to compare')
parser.add_argument('--wilcoxon_type', type=str, required=False, choices=['models', 'confidences'], help='Significance test type')
parser.add_argument('--wilcoxon_metric', type=str, required=False, choices=['f1', 'precision', 'recall'], help='Metric to use for models comparison')
parser.add_argument('--classification_report', type=str, required=False, help='Print classification report of an already tested experiment. '
                                                                              'Takes model version as an input')
parser.add_argument('--data_stats', action='store_true', required=False, help='Prints data statistics. Requires --data_dir')

args = parser.parse_args()

if args.split_data:
    if not args.data_dir:
        raise ValueError("Splitting data requires --data_dir argument to be passed")
    split_data(args.data_dir)

if args.create_vocab:
    try:
        start = time.time()
        create_vocab_file(args.create_vocab, args.embedding_dim)
        end = time.time()
        duration = round(end - start, 2)
    except ValueError as err:
        logger.error(f"Unable to create vocabulary. {err}")
        sys.exit(1)
    else:
        logger.info(f'Vocabulary created in {duration} seconds and file saved to {args.create_vocab}/vocab.txt')

if args.train:
    if not args.data_dir:
        raise ValueError("Passing --train expects --data_dir to be passed.")
    elif not args.model_dir:
        raise ValueError("Passing --train expects --model_dir to be passed.")

    vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    word2idx, idx2word, vocab_size = get_vocab(vocab_file)

    js_train_dataset = JSDataset(os.path.join(args.data_dir, 'train'), args.max_seq_len, args.embedding_type, word2idx)
    js_train_dataloader = DataLoader(js_train_dataset, batch_size=args.batch_size)

    js_dev_dataset = JSDataset(os.path.join(args.data_dir, 'dev'), args.max_seq_len, args.embedding_type, word2idx)
    js_dev_dataloader = DataLoader(js_dev_dataset, batch_size=args.batch_size)

    model_dir = args.model_dir

    embedding_dim = 768 if args.embedding_type == 'bert' else args.embedding_dim

    model_params = {
        'max_seq_len': args.max_seq_len,
        'embedding_dim': embedding_dim,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'embedding_type': args.embedding_type
    }

    logger.info(f'Training the model with the following parameters: {model_params}')

    word2vec = os.path.join(args.data_dir, 'word2vec')
    embedding = BERTEmbedding(vocab_file) if args.embedding_type == 'bert' \
        else Word2VecEmbedding(word2vec) if args.embedding_type == 'word2vec' \
        else nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    model = JSMalCatcherModel(word2idx, args.max_seq_len, embedding)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    now = datetime.now(timezone('US/Eastern'))
    base_version = f'v{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    model_version = f'{base_version}.{now.second:02d}' if os.path.isdir(f'{model_dir}/{base_version}') else base_version
    os.mkdir(os.path.join(model_dir, model_version))
    model_file = os.path.join(model_dir, model_version, 'model.pt')
    best_valid_loss = float('inf')
    best_valid_acc = float('-inf')

    metrics = {'epoch': [], 'train_acc': [], 'valid_acc': [], 'valid_loss': []}

    start = time.time()
    progress = tqdm(range(args.n_epochs))
    for epoch in progress:
        progress.set_description(f'Training')
        train_loss, train_acc = train(model, js_train_dataloader, optimizer, criterion)
        valid_loss, valid_acc, _ = evaluate(model, js_dev_dataloader, criterion)

        metrics['epoch'].append(epoch + 1)
        metrics['train_acc'].append(train_acc)
        metrics['valid_acc'].append(valid_acc)
        metrics['valid_loss'].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_file)

    end = time.time()
    duration = round(end - start, 2)

    logger.info(f'Finished training {args.n_epochs} epochs in {duration} seconds.')
    logger.info(f'Validation Accuracy: {best_valid_acc * 100:.2f}%.')
    logger.info(f'Model saved in {model_file}.')

    dataset_name = args.data_dir[args.data_dir.rindex("/") + 1:]

    runtime_metrics = {
        'timestamp': now,
        'duration': duration,
        'model_version': model_version,
        'valid_acc': best_valid_acc,
        'dataset': dataset_name
    }

    runtime_metrics.update(model_params)

    if not args.test and '--test' in sys.argv:
        model.load_state_dict(torch.load(model_file))
        model.eval()  # Put model in eval mode to turn off dropout regularization
        js_test_dataset = JSDataset(os.path.join(args.data_dir, 'test'), args.max_seq_len, args.embedding_type, word2idx)
        js_test_dataloader = DataLoader(js_test_dataset, batch_size=args.batch_size)
        logger.info('Testing saved model on test dataset. This may take a while...')
        test_loss, test_acc, metrics_dict = evaluate(model, js_test_dataloader, criterion, True)
        runtime_metrics.update({'accuracy': test_acc, 'precision': metrics_dict['precision'], 'recall': metrics_dict['recall'], 'f1': metrics_dict['f1']})
        predictions_csv = os.path.join(model_dir, model_version, 'predictions.csv')
        save_predictions(predictions_csv, metrics_dict)
        logger.info(f'Saved predictions in {predictions_csv}')
        logger.info(f'Test Accuracy: {test_acc * 100:.2f}%.')
        logger.info(f"Classification report: \n{metrics_dict['classification_report']}")

    # Save metrics (per epoch) and experiments CSV to track experiments
    metrics_csv = os.path.join(model_dir, model_version, 'metrics.csv')
    experiments_csv = os.path.join(model_dir, 'experiments.csv')
    if os.path.isfile(experiments_csv):
        experiments_df = pd.read_csv(experiments_csv)
    else:
        experiments_df = pd.DataFrame()
    experiments_df = experiments_df.append(runtime_metrics, ignore_index=True)
    experiments_df.to_csv(experiments_csv, index=False)
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(metrics_csv, index=False)

    logger.info(f'Model metrics in {metrics_csv}')
    logger.info(f'Experiments metrics updated in {experiments_csv}')

elif args.test:
    ####################################
    # Testing previously-trained model #
    ####################################

    if not args.model_dir:
        raise ValueError("Passing --test expects --model_dir")
    elif not args.data_dir:
        raise ValueError("Passing --test expects --data_dir")
    vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    model_dir = args.model_dir
    model_version = args.test
    model_file = os.path.join(model_dir, model_version, 'model.pt')
    if not os.path.isfile(model_file):
        raise ValueError(f"Model {model_file} does not exist.")
    word2idx, idx2word, vocab_size = get_vocab(vocab_file)
    embedding_dim = 768 if args.embedding_type == 'bert' else args.embedding_dim
    word2vec = os.path.join(args.data_dir, 'word2vec')
    embedding = BERTEmbedding(vocab_file) if args.embedding_type == 'bert' \
        else Word2VecEmbedding(word2vec) if args.embedding_type == 'word2vec' \
        else nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    model = JSMalCatcherModel(word2idx, args.max_seq_len, embedding)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    js_test_dataset = JSDataset(os.path.join(args.data_dir, 'test'), args.max_seq_len, args.embedding_type, word2idx)
    js_test_dataloader = DataLoader(js_test_dataset, batch_size=args.batch_size)
    criterion = nn.BCEWithLogitsLoss()

    logger.info(f'Testing model {model_version} with {args.embedding_type} embeddings')

    test_loss, test_acc, metrics = evaluate(model, js_test_dataloader, criterion, True)
    save_predictions(os.path.join(model_dir, model_version, 'predictions.csv'), metrics)
    logger.info(f'Test Accuracy: {test_acc * 100:.2f}%.')
    logger.info(f"Classification report: \n{metrics['classification_report']}")

    """
    Testing without training support is provided if we need to run ad-hoc test on new dataset using a model that was
    already trained. Therefore, experiments.csv is not updated in this case. Also, testing takes considerably less time
    than training so we can re-run the test experiment to reproduce the results.
    """

elif args.wilcoxon_pairs:
    if not args.wilcoxon_type:
        raise ValueError("Missing Wilcoxon test type. It can be either models or confidences")
    elif args.wilcoxon_type == 'models' and not args.wilcoxon_metric:
        raise ValueError("Required Wilcoxon metric to use is missing. When comparing models we either need to compare precision, recall or f1")
    elif "_" not in args.wilcoxon_pairs:
        raise ValueError("Pairs to compare needs to be in the following format: modelversion1_modelversion2 when comparing confidences, "
                         "or v1exp1,v1exp2_v2exp1,v2exp2 for comparing models")
    elif not args.model_dir:
        raise ValueError("Models directory is required for statistical significance testing")
    wilcoxon_pairs = args.wilcoxon_pairs
    wilcoxon_type = args.wilcoxon_type
    pairs_list = wilcoxon_pairs.split('_')
    first = pairs_list[0].split(',')
    second = pairs_list[1].split(',')

    assert len(first) == len(second), "Pairs to compare need to be of equal length"

    if wilcoxon_type == 'models':
        wilcoxon_metric = args.wilcoxon_metric
        experiments_df = pd.read_csv(os.path.join(args.model_dir, 'experiments.csv'))
        first_df = experiments_df[experiments_df.model_version.isin(first)]
        second_df = experiments_df[experiments_df.model_version.isin(second)]
        first_type = set(first_df.embedding_type.tolist())
        second_type = set(second_df.embedding_type.tolist())
        x = [float(i) for i in first_df[wilcoxon_metric].tolist()]
        y = [float(i) for i in second_df[wilcoxon_metric].tolist()]
        wilcoxon_result = wilcoxon(x, y)
        logger.info(f"Compared {first_type} {first} with {second_type} {second} using {wilcoxon_metric} and p-value is {wilcoxon_result.pvalue}")
    elif wilcoxon_type == 'confidences':
        first_model = first[0]
        second_model = second[0]
        first_df = pd.read_csv(os.path.join(args.model_dir, first_model, 'predictions.csv')).sort_values(by='sample_id')
        second_df = pd.read_csv(os.path.join(args.model_dir, second_model, 'predictions.csv')).sort_values(by='sample_id')
        x = [float(i) for i in first_df.confidence.tolist()]
        y = [float(i) for i in second_df.confidence.tolist()]
        avg_x = statistics.mean(x)
        avg_y = statistics.mean(y)
        wilcoxon_result = wilcoxon(x, y)
        logger.info(f"Compared confidences for {first_model} of average {avg_x} with {second_model} of average {avg_y} and p-value is {wilcoxon_result.pvalue}")
elif args.classification_report:
    if not args.model_dir:
        raise ValueError('Classification report expects a model_dir argument')
    elif not os.path.isdir(args.model_dir):
        raise ValueError(f'Model directory {args.model_dir} does not exist.')
    elif not os.path.isdir(os.path.join(args.model_dir, args.classification_report)):
        raise ValueError(f'Model version {args.classification_report} does not exist in {args.model_dir}')
    preds_df = pd.read_csv(os.path.join(args.model_dir, args.classification_report, 'predictions.csv'))
    y = preds_df.y.tolist()
    y_hat = preds_df.y_hat.tolist()
    logger.info(f'Classification report for {args.classification_report} is \n{classification_report(y, y_hat)}')
    logger.info(f'Accuracy: {round(accuracy_score(y, y_hat) * 100, 2)}')
    logger.info(f'Precision: {round(precision_score(y, y_hat) * 100, 2)}')
    logger.info(f'Recall: {round(recall_score(y, y_hat) * 100, 2)}')
    logger.info(f'F1: {round(f1_score(y, y_hat) * 100, 2)}')
elif args.data_stats:
    if not args.data_dir:
        raise ValueError('--data_stats requires --data_dir argument')

    dataset_analytics(args.data_dir)

logger.info('Goodbye!')
