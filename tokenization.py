import os
import requests
from lxml import html
import re

global vocab


def get_data():
    cwd = os.getcwd()
    filename = "raw_data.txt"
    filepath = os.path.join(cwd, filename)
    headers = {"User-Agent": "MyApp/1.0 (himi.rockeveryone@gmail.com)"}

    url = "https://en.wikisource.org/wiki/The_Verdict"
    response = requests.get(url=url, headers=headers)

    tree = html.fromstring(response.content)

    # Extract paragraphs directly
    paragraphs = tree.xpath("//div[@id='mw-content-text']//p//text()")

    txxt = "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())

    with open(filepath, "w") as f:
        f.write(txxt)

    clean_txt = re.split(r'([,.:;?_!"()\']|--|\s)', txxt)

    result = list(set([r.strip() for r in clean_txt if r.strip()]))
    result.append("<|unk|>")
    vocab = {result[i]: i for i in range(len(result))}
    return vocab, txxt


class BaseTokenizer:
    """Base Level Word Token generator and decording it back into original form"""

    def __init__(self, corpus):
        self.str_to_int = corpus
        self.int_to_str = {st: ind for st, ind in self.str_to_int.items()}

    def encoder(self, txxt):
        """Create int to word mapping for model training"""
        clean_txt = re.split(r'([,.:;?_!"()\']|--|\s)', txxt)
        result = list(set([r.strip() for r in clean_txt if r.strip()]))
        tokens = [
            (
                self.str_to_int[result[i]]
                if result[i] in self.str_to_int
                else self.str_to_int["<|unk|>"]
            )
            for i in range(len(result))
        ]
        return tokens

    def decoder(self, ids):
        """Return the word back based on the integer return by the model"""
        txxt = " ".join([self.int_to_str[i] for i in ids])
        txxt = re.sub(r'\s+([,.?!"()\'])', r"\1", txxt)
        return txxt


import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class BaseGPTDataset(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        tokens_ids = tokenizer.encode(txt)
        print("LEN_TOKEN", len(tokens_ids))
        for i in range(0, len(tokens_ids) - max_length, stride):
            input_chunk = tokens_ids[i : i + max_length]
            target_chunk = tokens_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.input_ids)

    def __getitem__(self, index):
        """Return the dataset row using index"""
        return self.input_ids[index], self.target_ids[index]


def create_data_loader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    batch_size: context size of tokens
    max_length: max context size
    stride: distance between start index of two consecutive batches (keep it 1 for max overlap)
    shuffle:
    drop_last: drop the last batch if the batch_size is not conforming.
    num_workers:

    Returns: returns a Dataloader Object
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = BaseGPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


######

# Creating Embeddings


if __name__ == "__main__":
    vocab, raw_txt = get_data()
    base_token = BaseTokenizer(vocab)
    txt = """ Its the last he painted, you know, " Mrs. Gisburn said with pardonable pride
            """
    ids = base_token.encoder(txt)
    # print(ids)
    print("RAW_TEXT", raw_txt[:500])

    dataloader = create_data_loader_v1(
        raw_txt, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    dataloader = create_data_loader_v1(
        raw_txt, batch_size=8, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, target = next(
        data_iter
    )  # gives the input tensor and target tensor using sliding window
    print("INPUTS", inputs)
    print("NEXT_BATCH shape", inputs.shape)

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print("TOKEN INPUT SHAPE", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print("TOKEN EMBEDDINGS SHAPE", token_embeddings.shape)

    # GPT Absolute Embedding approach

    context_len = 4
    pos_embedding_layer = torch.nn.Embedding(
        context_len, output_dim
    )  # create embedding of size output_dim random weights for each batch
    pos_embeddings = pos_embedding_layer(
        torch.arange(context_len)
    )  # basicaly maps the vector representation in emebd layer {pos independent}

    print("POS_EMBEDDING: ", pos_embeddings)
    # print("POS_EMBEDDING_SHAPE: ", pos_embeddings.shape)

    # input_embedding = token_embeddings + pos_embeddings
    # print("INPUT EMBEDDINGS: ", input_embedding.shape)

    # query = inputs[1] # Using first batch of size 8 and 4 tokens row

    # attention_scores = torch.empty(inputs.shape[0]) # create a empty linear tensor of shape equal to batch_size=8

    # for i, input_emb in enumerate(inputs):
    #     attention_scores[i] = torch.dot(input_emb, query) #

    # print("ATTENTION_SCORES", attention_scores)

    # attn_wghts = attention_scores / attention_scores.sum()
    # print("ATTENTION_WEIGHTS", attn_wghts)

    # ex:
    vocab_size = 50257
    output_dim = 3
    txt1 = "Your Journey starts with one step"
    new_data_loader = create_data_loader_v1(
        txt1, batch_size=1, max_length=5, stride=1, shuffle=False
    )
    iter_new_dt = iter(new_data_loader)
    inputs, target = next(iter_new_dt)

    # torch.manual_seed(123)
    ex_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    ex_token_embedding = ex_embedding_layer(inputs[0])

    print("INPUT TOKEN SHAPE: ", inputs.shape)
    print("INPUT_TOKEN: ", inputs)

    print("TOKEN EMBEDDINGS :", ex_token_embedding)
    query = ex_token_embedding[1]
    print("QUERY", query)
    attention_scores = torch.empty(ex_token_embedding.shape[0])
    print("ATTENTION_SCORES_SHAPE: ", attention_scores.shape)
    for ind, inp in enumerate(ex_token_embedding):
        attention_scores[ind] = torch.dot(inp, query)

    attn_weight_tmp = attention_scores / attention_scores.sum()
    print("SUM WEIGHT: ", attn_weight_tmp.sum())

    # using pytorch implementation of softmax
    attention_scores = torch.softmax(attention_scores, dim=0)

    print("ATTENTION_SCORES", attention_scores)
    print("ATTENTION_WTS_SHAPE: ", attention_scores.shape)

    # Calculating context vector for token ~ 2
    query = ex_token_embedding[1]
    print("QUERY", query)
    context_vector = torch.zeros(query.shape)

    # Getting context vector by multiply attn wts and weights intialized
    for i, wt in enumerate(ex_token_embedding):
        context_vector += attention_scores[i] * wt

    print("CONTEXT VECTOR: ", context_vector)

    print("TOKEN_EMBEDDING_SHAPE: ", ex_token_embedding.shape)
    attn_scores = torch.empty(5, 5)
    for i, iwt in enumerate(ex_token_embedding):
        for j, jwt in enumerate(ex_token_embedding):
            attn_scores[i, j] = torch.dot(iwt, jwt)

    print("ALL scores: ", attn_scores)
    # OR matrix multiplication
    attn_scores = ex_token_embedding @ ex_token_embedding.T  # make 5x 5 matrix
    print("scores: ", attn_scores)

    normalized_weights = torch.softmax(attn_scores, dim=-1)
    print("new_normalized_wts: ", normalized_weights)

    all_context_vectors = normalized_weights @ ex_token_embedding
    print("ALL CONTEXT VECS: ", all_context_vectors)
