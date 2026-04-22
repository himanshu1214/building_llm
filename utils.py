import tiktoken
import torch
import builtins

real_print = print

def debug_print(*args, **kwargs):
    real_print(*args, **kwargs)

# mute global print
builtins.print = lambda *args, **kwargs: None

def text_to_token(text, tokenize):
    encoded = tokenize.encode(text, allowed_special={"<|endoftext|>"})  # python obj
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # convert into tensort
    return encoded_tensor


def token_ids_to_text(token_ids, tokenize):
    decoded_tensor = token_ids.squeeze(0)
    text = tokenize.decode(decoded_tensor.tolist())
    return text