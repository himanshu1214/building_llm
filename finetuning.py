import zipfile
import os
from pathlib import Path
import pandas as pd
import requests
from torch.utils.data import Dataset
import tiktoken
import torch
from torch.utils.data import DataLoader
from pretraining import evaluate_model

def download_url_contents():
    extract_folder = "sms_data"
    modified_folder_path = os.path.join(
        os.getcwd(), extract_folder, "SMSSpamCollection.tsv"
    )
    if os.path.exists(modified_folder_path):
        return
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    output_path = "sms_spam_collection.zip"

    extract_folder_path = os.path.join(os.getcwd(), extract_folder, "SMSSpamCollection")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Downloaded!")
    with zipfile.ZipFile("sms_spam_collection.zip", "r") as zip_ref:
        zip_ref.extractall("sms_data")
    os.rename(extract_folder_path, modified_folder_path)

    print("Extracted!")


def get_data(df):
    spam_records_record = df[df["Label"] == "spam"].shape[0]  # get the record
    non_spam_df = df[df["Label"] == "ham"].sample(spam_records_record, random_state=123)
    combin_df = pd.concat([df[df["Label"] == "spam"], non_spam_df])
    return combin_df


def split_data(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_ind = int(len(df) * train_frac)
    val_ind = train_ind + int(len(df) * validation_frac)
    train_df = df[:train_ind]
    val_df = df[train_ind: val_ind]

    test_df = df[val_ind:]
    return train_df, val_df, test_df



class SpamDataSet(Dataset):
    def __init__(self, file_path, max_len, tokenizer, pad_token_id=50256):
        # read data
        self.df = pd.read_csv(file_path)
        
        # encode data
        self.encoded_text = [tokenizer.encode(txt) for txt in self.df['text']]

        # Truncate messages > max_len
        if max_len is None:
            self.max_len = self._longest_encoded_ln()
        else:
            self.max_len = max_len
            self.encoded_text = [
                encoded_tx[:max_len] for encoded_tx in self.encoded_text
            ]

        # Padding smaller sequences to the longest sequence
        self.encoded_text = [
            encoded_txt + [pad_token_id]*
            (self.max_len - len(encoded_txt)) 
             for encoded_txt in self.encoded_text
        ]



    def __getitem__(self, key):
        """
        Convert the convert to return as List
        """
        encoded = self.encoded_text[key]
        label = self.df.iloc[key]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.df)

    def _longest_encoded_ln(self):
        mx_ln = float("-inf")
        for tokn in self.encoded_text:
            mx_ln = max(mx_ln, len(tokn))

        return mx_ln

## Load Pre-trained model

# Add Classification layer
def calc_accuracy_loader(model, data_loader, device, num_batches=None):
    """
    Calculating classification accuracy
    """
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)

    else: num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else: break

    # print("CORRECT", correct_predictions)
    # print("NUM EX: ", num_examples)
    return correct_predictions / num_examples


def cal_loss_batch(input_batch, target_batch, device, model):
    """
    Defines a new loss function
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# Calculate classification loss
def calculate_loss_loader(data_loader, num_batches, model, device):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    

    for i, (input_data, target_data) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch=input_data, target_batch=target_data, device=device, model=model)
            total_loss += loss.item()

        else: break
    return total_loss / num_batches

def evaluate_classfication_mode(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(model=model, data_loader=train_loader, num_batches=eval_iter, device=device)
        test_loss = calculate_loss_loader(model=model, data_loader=val_loader, num_batches=eval_iter, device=device)

    model.train()
    return train_loss, test_loss

# Finetuning model for classification
def train_classifier(num_epocs, optimizer, train_loader, val_loader, device, eval_iter, model, eval_freq):
    """ this function is used to train the data first """
    visited, global_step = 0, -1
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    for epoch in range(num_epocs):
        model.train()

        for input_batch, target_batch in train_loader: 
            optimizer.zero_grad()
            loss_cal = cal_loss_batch(input_batch=input_batch, target_batch=target_batch, model=model, device=device)
            loss_cal.backward() # gradient calculation
            optimizer.step() # updates the model weights
            visited += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classfication_mode(
                    model=model, train_loader=train_loader, val_loader=val_loader, eval_iter=eval_iter, device=device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch + 1} Step : {global_step:06d} : "
                        f"Train loss {train_loss*100}%"
                        f"Val loss {val_loss*100}%"
                        )
        train_accuracy = calc_accuracy_loader(data_loader=train_loader, model=model, device=device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(data_loader=val_loader, model=model, device=device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy}")
        print(f"Val accuracy: {val_accuracy}")
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

    return train_losses, val_losses, train_acc, val_acc, visited

import matplotlib.pyplot as plt
# Plot classifier
def plot_losses(epoc, visited, train_val, val_val, label):

    # 2 plots on the same axis
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoc, train_val, label=f'Training {label}')
    ax1.plot(epoc, val_val, linestyle="-.", label=f'Validation {label}')


    #x , y labeling
    ax1.set_xlabel("Epocs")
    ax1.set_ylabel(f"{label.capitalize()}")
    ax1.legend()

    ax2 = ax1.twiny() # 2nd x-xaxis
    ax2.plot(visited, train_val, alpha=0)
    ax2.set_xlabel("Visited")

    fig.tight_layout()
    plt.savefig(f"{ label }-plot.pdf")
    plt.show()

# Classify new reviews:
def classify_review(text, model, tokenizer, device, pad_token=50256, max_len=None):
    model.eval()
    tokenized_input = tokenizer.encode(text)
    context_len = model.pos_emb.weight.shape[0]
    tokenized_input = tokenized_input[:min(max_len, context_len)]
    tokenized_input += [pad_token]*(max_len - len(tokenized_input))
    tokenized_tensor = torch.tensor(tokenized_input, device=device).unsqueeze(0) # add batch dimension
    with torch.no_grad():
        logits = model(tokenized_tensor)[:, -1, :]
    pred_label = torch.argmax(logits, dim=-1).item()

    return "spam" if pred_label == 1 else "not spam"

if __name__ == "__main__":
    download_url_contents()
    extract_folder = "sms_data"
    modified_folder_path = os.path.join(
        os.getcwd(), extract_folder, "SMSSpamCollection.tsv"
    )
    df = pd.read_csv(
        modified_folder_path, sep="\t", header=None, names=["Label", "text"]
    )

    print(df)

    balanced_df = get_data(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, val_df, test_df = split_data(balanced_df, 0.7, 0.1)
    cwd = os.getcwd()
    def get_fullfile_path(filename):
        return os.path.join(cwd, "sms_data", filename)
    train_df.to_csv(get_fullfile_path("train.csv"), index=False)
    val_df.to_csv(get_fullfile_path("val.csv"), index=False)
    test_df.to_csv(get_fullfile_path("test.csv"), index=False)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_data = SpamDataSet(file_path=get_fullfile_path("train.csv"), max_len=None, tokenizer=tokenizer)
    val_data = SpamDataSet(file_path=get_fullfile_path("val.csv"), max_len=train_data.max_len, tokenizer=tokenizer)
    test_data = SpamDataSet(file_path=get_fullfile_path("test.csv"), max_len=train_data.max_len, tokenizer=tokenizer)

    # Add PyTorch Data Loaders
  

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_data, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=False)
    test_loader = DataLoader(dataset=test_data, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=False)
    
    # for inp_batch, target_batch in train_loader:
    #     print("INP: ", inp_batch.shape)
    #     print("TARGET: ", target_batch.shape)

    print("TRAIN DATA: ", len(train_loader))
    print("VAL DATA: ", len(val_loader))
    print("TEST DATA: ", len(test_loader))


    # Add model config for gpt2
    model_selected = "gpt2-small-124M"
    input_prompt = "every effort moves"

    base_config = {"vocab_size": 50257, 
                   "context_length": 1024, 
                   "drop_rate": 0.0,
                   "qkv_bias": True}
    
    from model_configs import model_configs_map
    base_config.update(model_configs_map[model_selected])

    print("BASIC CONFIG: ", base_config)

    # Loading a pre build model

    from gpt_download import download_and_load_gpt2
    from gpt_model import GPTModel
    from load_gpt2_model import load_weight

    model_size = model_selected.split("-")[-1]

    # gets the GPT2 model weights
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # Load the GPT2 configs
    model = GPTModel(base_config)

    # loading the model weights
    load_weight(model, params)
    model.eval()

    test_txt = "Every efforts moves you"
    from gpt_model import generate_model_text
    from utils import text_to_token, token_ids_to_text

    token_ids = generate_model_text(
        model=model,
        x=text_to_token(test_txt, tokenize=tokenizer),
        max_new_tokens=15,
        context_ln=base_config["context_length"]
    )

    print("ThE output: ", token_ids_to_text(token_ids=token_ids, tokenize=tokenizer))

    ## test Spam Detector on the model

    text_2 = "Is the following text 'spam' ? Answer with 'Yes' or 'No'"

    for param in model.parameters():
        param.requires_grad = False

    # Add classification layer

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=base_config["emb_dim"],
        out_features=num_classes
    )

    for param in model.trans_blck[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    
    # Changing the output layer from 50270 to 2 classes
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs dim: ", inputs.shape)

    # encoded IDs
    with torch.no_grad():
        outputs = model(inputs)

    print("Output shape: ", outputs.shape) # now ouput shape is (1, 4, 2)

    probas = torch.softmax(outputs[:, -1, :], dim=-1) # picking the last token (1, 2)
    label = torch.argmax(probas) # picking the one with largest prob
    print("PRED LABEL: ", label.item())

    logits = outputs[:, -1, :]
    print("test-logit-shape: ", logits.shape)
    label = torch.argmax(logits)
    print("PRED LABEL -no-softmax", label.item())

    # get device
    device = torch.device("cuda")
    model.to(device)

    torch.manual_seed(123)

    train_accuracy = calc_accuracy_loader(
        data_loader=train_loader, model=model, device=device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
         data_loader=val_loader, model=model, device=device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
         data_loader=test_loader, model=model, device=device, num_batches=10
    )

    print("Train accuracy: ", train_accuracy*100, "%")
    print("Val accuracy: ", val_accuracy*100, "%")
    print("Test accuracy: ", test_accuracy*100, "%")


    # Call to Loss Loader
    with torch.no_grad():
        train_loss = calculate_loss_loader(model=model, data_loader=train_loader, device=device, num_batches=5)
        val_loss = calculate_loss_loader(model=model, data_loader=val_loader, device=device, num_batches=5)
        test_loss = calculate_loss_loader(model=model, data_loader=test_loader, device=device, num_batches=5)

    
    print("Train loss: ", train_loss)
    print("Val loss: ", val_loss)
    print("Test loss: ", test_loss)

    import time
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epocs = 5


    train_losses, val_losses, train_acc, val_acc, visited = train_classifier(
        model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, device=device, eval_iter=5, eval_freq=50,
        num_epocs=num_epocs
    )

    end_time = time.time()
    print("total_time in training: ", (end_time - start_time) / 60, "minutes")

    # LOSS PLOTS
    epocs_tensor = torch.linspace(0, num_epocs, len(train_losses))
    visited_tensor = torch.linspace(0, visited, len(train_losses))
    plot_losses(epocs_tensor, visited_tensor, train_losses, val_losses, "loss")

    #ACCURACY PLOTS
    epocs_tensor = torch.linspace(0, num_epocs, len(train_acc))
    visited_tensor = torch.linspace(0, visited, len(train_acc))
    plot_losses(epocs_tensor, visited_tensor, train_acc, val_acc, "accuracy")


    # Test independent reviews

    text_1 = ("You are a winner you have been specially"
              "selected to receive $1000 cash or $2000 award")
    
    print("CLASSIFY REVIEW TXT 1: ", classify_review(text_1, model, tokenizer, device, max_len=train_data.max_len))

    text2 = (
        "Hey, just want to check if we are still on"
        "for dinner tonight? Let me know"
    )
    
    print("CLASSIFY REVIEW TXT 2: ", classify_review(text2, model, tokenizer, device, max_len=train_data.max_len))

    torch.save(model.state_dict(), "review_classifier.pth")

    # Load model
    model_state_dict = torch.load("review_classifier.pth", map_location=device)
    model.load_state_dict(model_state_dict)