# ML Model

Pytorch-based model for detecting malicious JavaScript. Assumes preprocessed files produced by [js-mal-catcher](https://github.com/crazyeights225/js-mal-catcher).

## Splitting Data

Takes an input the path to text files. Splits them to 60% for training, 20% for development and 20% for testing. New `train`, `dev`, `test` folders are created
as a result of this operation.

```commandline
python ml/main.py --split_data --data_dir /path/to/txt/files
```

## Create vocabulary and Word2Vec model

This creates the vocabulary and word2vec embeddings using the training corpus. This is run once per dataset. After running, we end up with `vocab.txt`
and `word2vec` model. The point of word2vec is that it initializes the embeddings using word2vec instead of the default PyTorch behaviour embeddings which are
initialized randomly.

```commandline
python ml/main.py --create_vocab /path/to/dataset_folder
```

## Train

Training will create a sub-folder in `model_dir` named as the model version. The main `model_dir` will contain the
`experiments.csv` and the sub-folder will contain `metrics.csv`, `predictions.csv` and `model.pt`. Metrics CSV includes stats per epoch such as validation
accuracy.

Model versions are based on experiment start timestamp in EST. For example `v11200735` ran on November 20 at 07:35 AM. In the unlikely event that two
experiments ran within a minute, `.second` is appended to the version.

`data_dir` folder needs to contain `train`, `dev` and `test` folders with every sample ending in either `_ATTACK.txt`
or `_BENIGN.txt`

Data samples are one API call per line for API calls-based embeddings or one bytecode instruction per line for bytecode.

```commandline
python ml/main.py  --model_dir /path/to/desired/model/output/dir --data_dir /path/to/data/dir --train --batch_size=32
```

Optionally (and recommended), you can also provide the `--test` flag without any arguments to test the newly trained model on the test dataset. Model
predictions and confidence are stored in `predictions.csv` in the model version directory.

### Full list of configurable parameters

| Parameter | Default | Notes |
|---|---|---|
|`batch_size`|32|How many samples the model sees at once.|
|`n_epochs`|5|How many training iterations.|
|`max_seq_len`|128|Sequences need to be of fixed size. Sequences containing more tokens than `max_seq_len` are truncated. Those containing less are padded with zeros.|
|`lr`|0.001|[Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer learning rate.|
|`embedding_type`|default|Supports default (random PyTorch embedding initialization), word2vec or bert.|
|`embedding_dim`|100|Vectors dimensions. Not configurable for bert (can only be 768). If using word2vec, needs to match dimensions the word2vec model is trained with.|

# Test

Testing outside of training is also supported to provide the ability to test ad-hoc or newly found datasets using previously-trained models. For example, the
following command tests model version `v1120735` that exists in
`/path/to/models/dir`using the `test` folder in `/path/to/data/dir`. Model predictions and confidence are stored in `predictions.csv` in the model version
directory.

```commandline
python ml/main.py --model_dir /path/to/models/dir --data_dir /path/to/data/dir --test v1120735
```

# Statistical Significance

Compares two pairs of experiments to decide whether there's statistical significance between the two. Supports two types of comparisons:

## Models

Compares two sets of model experiments. For example 10 experiments of word2vec with 10 experiments of bert. The two sets are separated by `_` and the
experiments are comma-separated. Compares the experiments using either precision, recall or f1.

```commandline
python ml/main.py --model_dir /path/to/models --wilcoxon_type models --wilcoxon_metric f1 --wilcoxon_pairs v11241711,v11241714_v1128820,v1128823
```

## Confidence

Compares two models for their confidences on the predictions. For example:

```commandline
python ml/main.py --model_dir /path/to/models --wilcoxon_type confidences --wilcoxon_pairs v1128823_v1128820
```

# Data analytics

Prints out some statistics about the data such as the malicious:benign ratio, total files in each train/dev/test split and average sequence length.

```commandline
python ml/main.py --data_dir /path/to/data --data_stats
```

# Resources

[BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#why-bert-embeddings)
