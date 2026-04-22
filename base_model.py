import torch
import torch.nn as nn
import gpt_config as cfg

class DummyGPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_embedding = nn.Dropout(cfg["context_length"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])) # Transformer Block
        self.final_norm = nn.Linear(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input):
        batch_size, sequence_ln = input.shape
        token_emb = self.token_embedding(input)
        pos_embedding = self.pos_embedding(torch.arange(sequence_ln), device= input.device)
        stack_embedding = token_emb + pos_embedding
        stack_drop_embedding = self.drop_embedding(stack_embedding)


class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

