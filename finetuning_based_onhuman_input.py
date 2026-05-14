import json
import os
import time
import urllib

import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from gpt_download import download_and_load_gpt2
from gpt_model import GPTModel
from model_configs import model_configs_map
from pretraining import (generature_with_decoding_strat, loss_loader,
                         plot_losses, train_model)
from utils import text_to_token, token_ids_to_text


# Loading Data
def download_instruction():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    response = requests.get(url)
    response.raise_for_status()

    if not os.path.exists("instructions_data.json"):
        with open("instructions_data.json", "wb") as f:
            f.write(response.content)


# Adding prompt functionalitys
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction: \n {entry['instruction']}"
    )
    input_text = f"\n\n### Input: \n {entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# Implementation Instruction based dataset class
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_data = []
        for entry in data:
            instruct_data = format_input(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            comb = instruct_data + response
            self.encoded_data.append(tokenizer.encode(comb))

    def __getitem__(self, key):
        return self.encoded_data[key]

    def __len__(self):
        return len(self.data)


# Custom collate func
def custom_collate_fn(
    batch, device, ignore_index=-100, pad_token_id=50256, allowed_max_ln=None
):
    """
    Helper function for padding the input based on the max batch input lenght
    and replace the target padding token except the last pad-token with placeholder
    to skip from training loss
    """
    batch_max_ln = max(len(item) + 1 for item in batch)
    input_ls = []
    target_ls = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_ln - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()  # create a tensor with indices
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # skip the first padding token

        if allowed_max_ln is not None:
            inputs = inputs[:allowed_max_ln]
            targets = targets[:allowed_max_ln]

        input_ls.append(inputs)
        target_ls.append(targets)

    # gen input / target tensor
    input_tensor = torch.stack(input_ls).to(device)
    target_tensor = torch.stack(target_ls).to(device)
    return input_tensor, target_tensor


# Fine tuning pre-trained model

# Gen test set responses

# Query local Ollama model


# Evaluating instruction fine-tune model

if __name__ == "__main__":
    download_instruction()
    with open("instructions_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(format_input(data[50]))
    desired_output = f"\n\n### Response: \n{data[50]['output']}"
    print(desired_output)

    # Partition data
    train_count = int(len(data) * 0.85)
    test_count = int(len(data) * 0.1)
    val_count = len(data) - train_count - test_count

    train_data = data[:train_count]
    test_data = data[train_count : train_count + test_count]
    val_data = data[train_count + test_count :]

    input_1 = [0, 1, 2, 3, 4]
    input_2 = [5, 6]
    input_3 = [7, 8, 9]

    batch = (input_1, input_2, input_3)
    from functools import partial

    custmized_collate_fn = partial(
        custom_collate_fn, device="cpu", allowed_max_ln=1024  # supported in gpt2
    )
    # Data Loaders initializer
    batch_size = 8
    num_workers = 4
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custmized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custmized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custmized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # Load pre-trained model
    base_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    model_select = "gpt2-medium-355M"
    model_config = model_configs_map[model_select]
    base_config.update(model_config)
    model_size = model_select.split("-")[-1]
    print("MODEL_SIZE: ", model_size)

    settings, params = download_and_load_gpt2(model_size, "gpt2")
    model = GPTModel(base_config)
    model.eval()

    torch.manual_seed(123)
    input_text = format_input(test_data[0])

    print("INPUT-TEXT: ", input_text)

    token_ids = generature_with_decoding_strat(
        model=model,
        batch_emb=text_to_token(input_text, tokenizer),
        max_new_tokens=35,
        context_ln=base_config["context_length"],
        top_k=25,
        temperature=1,
        eos_id=50256,
    )

    generated_text = token_ids_to_text(token_ids=token_ids, tokenize=tokenizer)
    print("GENERATE_TEXT: ", generated_text)

    # Finetuning

    torch.manual_seed(123)
    model.to("cuda")
    with torch.no_grad():
        train_loss = loss_loader(train_loader, model, "cuda", num_batches=5)

        test_loss = loss_loader(test_loader, model, "cuda", num_batches=5)

    print("TRAIN-LOSS: ", train_loss)
    print("TEST-LOSS: ", test_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epocs = 2

    train_losses, test_losses, visited = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        "cuda",
        num_epocs=num_epocs,
        eval_freq=100,
        eval_iter=2,
        start_context=format_input(test_data[0]),
        tokenizer=tokenizer,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print("TOT time: ", total_time / 60, "min")

    # Plot
    epocs_tensor = torch.linspace(0, num_epocs, len(train_losses))
    plot_losses(epocs_tensor, visited, train_losses, test_losses)

    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_txt = format_input(entry)

        token_ids = generature_with_decoding_strat(
            model=model,
            batch_emb=text_to_token(input_txt, tokenizer).to("cuda"),
            max_new_tokens=35,
            context_ln=base_config["context_length"],
            top_k=25,
            temperature=1,
            eos_id=50256,
        )

        generated_text = token_ids_to_text(token_ids=token_ids, tokenize=tokenizer)
        print("GENERATE_TEXT: ", generated_text)

    from tqdm import tqdm

    # Generating test set responses
    for i, entry in tqdm(val_data, total=len(val_data)):
        input_txt = format_input(entry)

        token_ids = generature_with_decoding_strat(
            model=model,
            batch_emb=text_to_token(input_txt, tokenizer).to("cuda"),
            max_new_tokens=256,
            context_ln=base_config["context_length"],
            top_k=25,
            temperature=1,
            eos_id=50256,
        )

        generated_text = token_ids_to_text(token_ids=token_ids, tokenize=tokenizer)
