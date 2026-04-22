from tokenization import create_data_loader_v1
from tokenization import get_data
import torch
import os
import re
import tiktoken
import gpt_config
from gpt_model import GPTModel, generate_model_text
from utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

## initialize model
model = GPTModel(gpt_config.GPT_CONGIF_124M)
model.eval()


# Text Gen Loss

txt1 = "every effort moves"
txt2 = "I really like"
batch = []
tokenizer = tiktoken.get_encoding("gpt2")
token_txt1 = torch.tensor(tokenizer.encode(txt1))
token_txt2 = torch.tensor(tokenizer.encode(txt2))
batch.extend([token_txt1, token_txt2])
encoded_in_tensor = torch.stack(batch, dim=0)
print("ENCODED tensor -1: ", encoded_in_tensor)

target_batch = []
target1 = " effort moves you"
target2 = " really likes chocolate"
token_target1 = torch.tensor(tokenizer.encode(target1))
token_target2 = torch.tensor(tokenizer.encode(target2))
target_batch.extend([token_target1, token_target2])
target_out_tensor = torch.stack(target_batch, dim=0)



with torch.no_grad():
    logits = model(encoded_in_tensor)

# softmax doesnot change, it only scale the values
prob = torch.softmax(logits, dim=-1) # (batch, seqlen, vocab_size) -> (2, 3, 50257)
# model generates 3 token not 1, eventhough it gets 3 input
# for each token, model generates the next token
# ex: every --> effort(target)
# effort(input) -- > moves (target)
# moves (input) --> you (target)
print("PROBS-shape: ", prob.shape)
 
 # keep only lastdim (2, 3, 50257) --> (2, 3, 1) 
 # best token based on prob across each row
token_ids = torch.argmax(prob, dim=-1, keepdim=True) 
print("TOKEN IDs-shape: ", token_ids)

print("TARGET TOKEN: ", token_ids_to_text(target_out_tensor[0], tokenizer))
print("PRED OUTPUT TOKEN : ", token_ids_to_text(token_ids[0].flatten(), tokenizer))

#Log Probs
txt_ind1 = 0
target_prob1 = prob[txt_ind1, [0, 1, 2], target_out_tensor[txt_ind1]]
txt_ind2 = 1
target_prob2 = prob[txt_ind2, [0, 1, 2], target_out_tensor[txt_ind2]]
log_prob = torch.log(torch.cat((target_prob1, target_prob2))) # using torch.log_softmax is more stable
print("LOG-PRB: ", log_prob)
avg_log_mean_prob = torch.mean(log_prob)
print("MEAN LOG PROB: ", avg_log_mean_prob)


# CALCULATE CROSS ENTROPY 
### Step1 : Flatten the logits and target and then apply cross_entropy
logits_flat = logits.flatten(0, 1)
target_out_tensor_flat = target_out_tensor.flatten()
print("LOGITS flat: ", logits_flat.shape)
print("TARGET  TENSOR FLAT: ", target_out_tensor_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, target_out_tensor_flat)
print("LOSS-cal: ", loss)

## READ verdict text file from stored data
cwd = os.getcwd()
filename = "raw_data.txt"
filepath = os.path.join(cwd, filename)
_, data = get_data()

print("TOTAL chars: ", len(data))
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(data)
print("TOT tokens: ", len(tokens))
train_ratio = 0.9
train_test_indx = int(train_ratio*len(data))
train_data = data[:train_test_indx]
test_data = data[train_test_indx:]

print("TRAIN: ", train_data[:500])

train_data_loader = create_data_loader_v1(train_data, batch_size=2, 
                                          max_length=gpt_config.GPT_PRETRAIN_CONFIG_124M["context_length"], 
                                          stride=gpt_config.GPT_PRETRAIN_CONFIG_124M["context_length"], # no overlap 
                                          drop_last=True,
                                          shuffle=True,
                                          num_workers=0)

test_data_loader = create_data_loader_v1(test_data, batch_size=2, 
                                          max_length=gpt_config.GPT_PRETRAIN_CONFIG_124M["context_length"], 
                                          stride=gpt_config.GPT_PRETRAIN_CONFIG_124M["context_length"], # no overlap 
                                          drop_last=False,
                                          shuffle=False,
                                          num_workers=0)

# print("TRAIN_DATA_LOADER", iter(train_data_loader)[0])
print("TRAIN_DATA")
torch.manual_seed(123)
# val = iter(train_data_loader)
# print("NEXTL :", next(val))
for x, y in iter(train_data_loader):
    print("X: ", x.shape, "Y: ", y.shape)



def loss_per_batch_calc(input_batch, target_batch, model, device):
    """
    This func is used to compute the loss over the set of batches
    """
    input_batch = input_batch.to(device) # move input to same device/gpu
    target_batch = target_batch.to(device) # including target
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def loss_loader(data_loader, model, device, num_batches=None):
    """
    This function is used to iterate over all batches
    Calculate loss and average them """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (inp, target) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_per_batch_calc(inp, target, model, device)
            total_loss += loss.item() # summation losses
        
    return total_loss / num_batches # Average loss

device = torch.device("cuda")
model = GPTModel(gpt_config.GPT_PRETRAIN_CONFIG_124M)
model.to(device) # cuda support gpu
with torch.no_grad():  # disable gradient tracking
    train_loss = loss_loader(train_data_loader, model, device) # 
    test_loss = loss_loader(test_data_loader, model, device)

print("TRAIN-LOSS: ", train_loss)
print("TEST-LOSS: ", test_loss)
def train_model(
    model,
    train_data,
    test_data,
    optimizer,
    device,
    num_epocs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    token_seen, global_step = 0, -1
    train_losses, test_losses, track_token_seen = [], [], []
    for epoch in range(num_epocs):
        # model train
        model.train()
        # get input, target token from data loader
        for input_b, target_b in train_data:
            optimizer.zero_grad() # restore gradient to 0 for each iteration in a batch
            loss = loss_per_batch_calc(input_b, target_b, model, device)
            loss.backward() # calculate loss gradient
            optimizer.step() # update the model weights
            token_seen += input_b.numel()
            global_step += 1

            # evaluate model step
            if global_step % eval_freq == 0: # optional step
                train_loss, test_loss = evaluate_model(model, train_data, test_data, device, eval_iter)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            track_token_seen.append(token_seen)

            print(f"""EPOCH {epoch + 1} STEP {global_step:06d}: 
                  TRAIN LOSS {train_loss:.3f}, 
                  TEST LOSS {test_loss:.3f}
                    """)
        
        # generate text after each epoch
        gen_print_sample(model, tokenizer, device, start_context)
    return train_losses, test_losses, track_token_seen
        

def evaluate_model(model, train_loader, test_loader, device, eval_iter):
    """
    this utility function is triggered to get the losses based on the new model weights
    updated based on model training. 
    It helps understand the losses trend
    """
    model.eval()
    with torch.no_grad(): # disable gradient tracking during testing phase
        train_loss = loss_loader(train_loader, model, device, num_batches=eval_iter)
        test_loss = loss_loader(test_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, test_loss

def gen_print_sample(model, tokenizer, device, context):
    """
    this utility function is used to generate neww tokens
    """
    model.eval() # helps skip dropout
    context_ln = model.pos_emb.weight.shape[0]
    encoded = text_to_token(context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_model_text(model=model, x=encoded, max_new_tokens=50, 
                                        context_ln=context_ln)
        
    decoded_txt = token_ids_to_text(token_ids=token_ids, tokenize=tokenizer)
    print("DECODED-txt: ",  decoded_txt.replace('\n', ' '))
    model.train()

torch.manual_seed(123)

model = GPTModel(gpt_config.GPT_CONGIF_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epocs = 10
train_losses, test_losses, tokens_seen = train_model(
    model, train_data_loader, test_data_loader, optimizer, device, 
    num_epocs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer
)



def plot_losses(epocs_seen, tokens_seen, train_losses, test_losses):
    fig, axes = plt.subplots(figsize=(5, 3))
    axes.plot(epocs_seen, train_losses, label="Training Losses")
    axes.plot(epocs_seen, test_losses, label="Test Losses")
    axes.set_xlabel("EPOCS")
    axes.set_ylabel("LOSS")
    axes.legend(loc="upper right")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = axes.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Token Seen")
    fig.tight_layout()
    plt.show()

epocs_tensor = torch.linspace(0, num_epocs, len(train_losses))
plot_losses(epocs_tensor, tokens_seen, train_losses, test_losses)

# Modifying txt generation func:
# Adding more decoding strategies ex: temperature scaling, top-k sampling
model = GPTModel(gpt_config.GPT_CONGIF_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

# Fix different device issue
# Get model device
device = next(model.parameters()).device
x = text_to_token("Every effort moves you ", tokenizer).to(device)
# move text to same model device
token_gen = generate_model_text(model=model, x=x, 
                                max_new_tokens=25, 
                                context_ln=gpt_config.GPT_CONGIF_124M["context_length"])

debug_print("TXT generated: ", token_ids_to_text(token_gen, tokenizer))

## Decoding strategies
# Temperature Scaling

# Top-k

def generature_with_decoding_strat(model, batch_emb, max_new_tokens ,context_ln, temperature, top_k, eos_id=None):
    for _ in range(max_new_tokens):
        batch_emb = batch_emb[:, -context_ln:] #
        with torch.no_grad():
            logits = model(batch_emb)
        logits = logits[:,-1, :] # get the last index / predicted token
        topk_logits, _ = torch.topk(logits, top_k) # filter using topk sampling
        min_val = topk_logits[:, -1]
        if top_k is not None:
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float('-inf')).to(logits.device), 
                            logits)


        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            emb_next = torch.multinomial(probs, num_samples=1)

        else:
            emb_next = torch.argmax(logits, dim=-1, keepdim=True)

        if emb_next == eos_id:
            break
            
        batch_emb = torch.cat((batch_emb, emb_next), dim=1)
    return batch_emb

# 
torch.manual_seed(123)


# Fix different device issue
# Get model device
model.eval() # turn on model evaluation , skips dropouts 
device = next(model.parameters()).device
x = text_to_token("Every effort moves you ", tokenizer).to(device)
token_ids = generature_with_decoding_strat(
    model, 
    batch_emb=x, 
    max_new_tokens=15, 
    context_ln=gpt_config.GPT_CONGIF_124M["context_length"],
    top_k=25,
    temperature=1.4
)

debug_print("GENRATED TXT BASED ON TEMP SCALING AND TOPK STRAT: ", token_ids_to_text(token_ids, tokenizer))

# Dictionary mapping each state to its model params
torch.save(model.state_dict(), "model.pth")

model = GPTModel(gpt_config.GPT_CONGIF_124M)
model.load_state_dict("model.pth", map_location=device)
model.eval()

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "model_and_optimizer_comb.pth")

# LOADING
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(gpt_config.GPT_CONGIF_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()


## LOADING PRETRAINED GPT-2 model
def load_wts_into_gpt(gpt, params):
    pass