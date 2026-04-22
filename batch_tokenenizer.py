import tiktoken
import torch
from base_model import DummyGPTModel
from gpt_config import GPT_CONGIF_124M

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Lets begin language model journey"
txt2 = "Each day brings joy of a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
stack_batch = torch.stack(batch, dim=0)

print(stack_batch)


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONGIF_124M)
