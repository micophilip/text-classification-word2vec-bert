import numpy as np
import torch
from gensim.models import word2vec
from transformers import BertTokenizer, BertModel

from dataset import get_vocab


class Word2VecEmbedding:
    def __init__(self, word2vec_file: str):
        model = word2vec.Word2Vec.load(word2vec_file)
        embedding_dim = model.vector_size
        model.wv["<pad>"] = np.full((1, embedding_dim), 0)
        model.wv["<unk>"] = np.random.rand(embedding_dim)
        self.model = model.wv
        self.key_to_idx = model.wv.key_to_index
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_word2vec_embeddings(self, text):
        embedded = [[self.model[word] for word in sentence.split(' ')] for sentence in text]
        embedded = torch.FloatTensor(np.array(embedded)).to(self.device)
        embedded = torch.permute(embedded, [1, 0, 2])
        return embedded


class BERTEmbedding:

    def __init__(self, vocab_file: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        word2idx, idx2word, vocab_size = get_vocab(vocab_file)
        all_vocab = list(word2idx.keys())
        special_tokens = all_vocab[:2]
        vocab_tokens = all_vocab[2:]
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.tokenizer.add_tokens(vocab_tokens)

        self.embedding_dim = 768

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(self.device)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert.eval()

    def get_bert_embeddings(self, text):
        tokenized_texts = [self.tokenizer.tokenize(sentence) for sentence in text]

        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
        segments_ids = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]

        token_tensor = torch.tensor(indexed_tokens).to(self.device)
        segment_tensor = torch.tensor(segments_ids).to(self.device)

        with torch.no_grad():
            outputs = self.bert(token_tensor, segment_tensor)
            hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            second_to_last_layer = token_embeddings[11]
            embedding_tensor = torch.permute(second_to_last_layer, [1, 0, 2])
            return embedding_tensor
