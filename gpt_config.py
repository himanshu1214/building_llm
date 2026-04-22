GPT_CONGIF_124M = {
    "vocab_size": 50257,  # total corpus size
    "context_length": 1024,  # number of tokens at a time used
    "emb_dim": 768,  # embedding dimension , transforming the input token into 768 dimensional vector
    "n_heads": 12,  # number of attention heads
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT_PRETRAIN_CONFIG_124M = {
    "vocab_size": 50257,  # total corpus size
    "context_length": 256,  # number of tokens at a time used
    "emb_dim": 768,  # embedding dimension , transforming the input token into 768 dimensional vector
    "n_heads": 12,  # number of attention heads
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
