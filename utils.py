import tiktoken
import torch
import builtins


def text_to_token(text, tokenize):
    encoded = tokenize.encode(text, allowed_special={"<|endoftext|>"})  # python obj
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # convert into tensort
    return encoded_tensor


def token_ids_to_text(token_ids, tokenize):
    decoded_tensor = token_ids.squeeze(0)
    text = tokenize.decode(decoded_tensor.tolist())
    return text
