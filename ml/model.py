from typing import Union

import torch
import torch.nn as nn

from embeddings import BERTEmbedding, Word2VecEmbedding

"""
SOURCE DISCLAIMER:
Used the following notebook as a starting guide:
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
"""


class JSMalCatcherModel(nn.Module):
    def __init__(self, word2idx: dict, max_seq_len, embedding_obj: Union[nn.Embedding, BERTEmbedding, Word2VecEmbedding]):
        super().__init__()

        hidden_dim = 200
        output_dim = 1
        n_layers = 2
        bidirectional = True
        dropout = 0.5
        embedding_dim = embedding_obj.embedding_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_type = 'bert' if embedding_obj.__class__.__name__ == 'BERTEmbedding' \
            else 'word2vec' if embedding_obj.__class__.__name__ == 'Word2VecEmbedding' \
            else 'default'

        self.embedding = embedding_obj.get_bert_embeddings if self.embedding_type == 'bert' \
            else embedding_obj.get_word2vec_embeddings if self.embedding_type == 'word2vec' \
            else embedding_obj

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        if self.embedding_type == 'default':
            encoded_texts = [[self.word2idx[word] for word in sentence.split(' ')] for sentence in text]

            text_tensor = torch.LongTensor(encoded_texts).to(self.device)

            # Our model expects shape in sent len x batch size. Permute tensor.
            text_tensor = torch.permute(text_tensor, [1, 0])

            embedded = self.embedding(text_tensor)
        else:
            embedded = self.embedding(text)

        # embedded = [file contents len, batch size, emb dim]
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[2, :, :], hidden[3, :, :]), dim=1))

        return self.fc(hidden)
