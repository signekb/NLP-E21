from typing import List
from datasets import load_dataset
import gensim.downloader as api


# DATASET
dataset = load_dataset("conllpp")
train = dataset["train"] # subset train part of dataset

# inspect the dataset
#print(train["tokens"][:1])
#print(train["ner_tags"][:1])
num_classes = train.features["ner_tags"].feature.num_classes

# CONVERTING EMBEDDINGS
import numpy as np

import torch

model = api.load("glove-wiki-gigaword-50")

from embedding import gensim_to_torch_embedding

# convert gensim word embedding to torch word embedding
embedding_layer, vocab = gensim_to_torch_embedding(model)

# # PREPARING A BATCH

def tokens_to_idx(tokens, vocab=model.key_to_index):
    """
    vocab("UNK"): Adds a token "UNK" for unknown tokens. So when encountering tokens not in vocab, 
    the word embedding for UNK will be used

    Takes in list of tokens

    Return the idx of the word, unless the word is unknown, then return the idx for "UNK"

    Ideas to understand this function:
    - Write documentation for this function including type hints for each argument and return statement
    - What does the .get method do?
        The get() method returns the value of the item with the specified key.
        Vocab is a dictionary, a
        I think it "gets" the number of the embedding of the current token

    - Why lowercase?
        Because everything is lowercased in vocab
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]

# sample batch of 10 sentences
batch_tokens = train["tokens"][:10] # 10 first sentences
batch_tags = train["ner_tags"][:10] # 10 first ner_tags
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens]
batch_size = len(batch_tokens)

# # compute length of longest sentence in batch
batch_max_len = max([len(s) for s in batch_tok_idx])

# prepare a numpy array with the data, initializing the data with 'PAD'
# and all labels with -1; initializing labels to -1 differentiates tokens
# with tags from 'PAD' tokens
batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len))
batch_labels = -1 * np.ones((batch_size, batch_max_len)) # because  the padding does not have a label, we just call the label -1 to indicate that it is padding.

# # copy the data to the numpy array
for i in range(batch_size):
    tok_idx = batch_tok_idx[i]
    tags = batch_tags[i]
    size = len(tok_idx)

    batch_input[i][:size] = tok_idx # the rest, after size of the sentence will then be padding
    batch_labels[i][:size] = tags # the rest, after size of the sentence will then be padding


# since all data are indices, we convert them to torch LongTensors
batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(batch_labels)

# CREATE MODEL
from LSTM import RNN

model = RNN(
    embedding_layer=embedding_layer, output_dim=num_classes + 1, hidden_dim_size=256
)

# FORWARD PASS
X = batch_input
y = model(X)

loss = model.loss_fn(outputs=y, labels=batch_labels) # cross entropy loss - with -1's removed!
# loss.backward()