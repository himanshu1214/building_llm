import gpt_config
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from multihead_llm import MultiHeadAttention

class DummyTransformerBlock(nn.Module):
    def __init__(self, gpt_config):
        pass

    def feed_forward(self):
        pass

class DummyLayerNorm(nn.Module):
    def __init__(self, gpt_config):
        pass

    def feed_forward(self):
        pass

class DummyGPTModel(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        self.token_emb = nn.Embedding(gpt_config["vocab_size"], gpt_config["emb_dim"])
        self.pos_emb = nn.Embedding(gpt_config["context_length"], gpt_config["emb_dim"])
        self.drop_emb = nn.Dropout(gpt_config["drop_rate"])
        self.trf_block = nn.Sequential(*[DummyTransformerBlock(gpt_config) for _ in range(gpt_config["n_layers"])])
        self.final_norm = DummyLayerNorm(gpt_config["emb_dim"])
        self.out_head = nn.Linear(gpt_config["emb_dim"], gpt_config["vocab_size"], bias=False)

    def forward(self, token_batch):
        batch_size, seq_ln = token_batch.shape
        token_emb = self.token_emb(token_batch) # converts into token embedding (batch_size, context_len, emb_dim)
        pos_embs = self.pos_emb(torch.arange(seq_ln, device=token_batch.device)) # gives context_ln x emb_dim
        x = token_emb + pos_embs
        x = self.drop_emb(x)
        x = self.trf_block(x) # capturing a list of transformer blocks in sequence block1 --> block2 --> block3
        x = self.final_norm(x) # scaling / normalizing
        logits = self.out_head(x) # dot product of x (batch_size, context_ln, emb_dim) weights (context_ln, emb_dim)
        return logits 


batch = []
txt1 = "Keep move with new world"
txt2 = "Can we get predict next"
tiktokenizer = tiktoken.get_encoding("gpt2")
batch.append((torch.tensor(tiktokenizer.encode(txt1))))
batch.append((torch.tensor(tiktokenizer.encode(txt2))))
batch = torch.stack(batch, dim=0) # shape (2, 6, 3)
print("BATCH shape: ", batch.shape, "\n")
print("BATCH : ", batch, "\n")


# model = DummyGPTModel(gpt_config.GPT_CONGIF_124M)
# logits = model(batch)
# print("logit shape: ", logits.shape)
# print("LOGITS: ", logits)

torch.manual_seed(123)
t_batch = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(t_batch)
print("tst output shape: ", output.shape)
print("TST OUT: ", output)

mean = output.mean(dim=-1, keepdim=True)
var = output.var(dim=-1, keepdim=True)

print("VAR: ", var, "\nMEAN: ", mean)

output = (output - mean)/torch.sqrt(var)
mean = output.mean(dim=-1, keepdim=True)
var = output.var(dim=-1, keepdim=True)

print("NORMALIZED : ", output)
print("normal mean: ", mean)
print("normal var: ", var)

class LayerNorm(nn.Module):
    """
    Normalizing token across embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable params (emb, )
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable params (emb, )

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) 
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps) 
        return self.scale*norm_x + self.shift # each token scaled based on the normalized vector hence some token dominate over other

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * ( 1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)

y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} Activate Func")
    plt.xlabel("X")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

class FeedForward(nn.Module):
    """
    This class takes the token with ~768 emb_dim which expanded to 4 times 
    space GELU activation helps optimization of params
    """
    def __init__(self, gpt_config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(gpt_config["emb_dim"], 4*gpt_config["emb_dim"]), # expanded to 4 time dimension
            GELU(), 
            nn.Linear(4*gpt_config["emb_dim"], gpt_config["emb_dim"]), # shrinking back to original dimension
        )

    def forward(self, x):
        return self.layers(x)

class ExampleDNN(nn.Module):

    def __init__(self, layer_count, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
          nn.Sequential(nn.Linear(layer_count[i], layer_count[i +1]), GELU())  for i in range(5)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # Shorting layer fixes the vanishing gradient problem
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x
    
layer_ct = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([1., 0., 1.])
torch.manual_seed(123)

model_without_shortcut = ExampleDNN(layer_count=layer_ct, use_shortcut=False)


def print_gradient(model, x):
    out = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(out, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(name, " gradient MEAN of ", param.grad.abs().mean().item())


print(print_gradient(model_without_shortcut, sample_input), "\n")

model_with_shortcut = ExampleDNN(layer_count=layer_ct, use_shortcut=True)
print(print_gradient(model_with_shortcut, sample_input))

class TransformerBlock(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=gpt_config["emb_dim"], 
            d_out=gpt_config["emb_dim"],
            context_ln=gpt_config["context_length"],
            num_heads=gpt_config["n_heads"],
            dropout=gpt_config["drop_rate"],
            qkv_bias=gpt_config["qkv_bias"]
        )

        self.ff = FeedForward(gpt_config) # for optimization of params during training 
        # Adding 2 LayerNorm
        self.norm1 = LayerNorm(gpt_config["emb_dim"]) #shape (emb_dim, )
        self.norm2 = LayerNorm(gpt_config["emb_dim"])
        self.drop_shortcut = nn.Dropout(gpt_config["drop_rate"])

    def forward(self, x):
        shortcut = x # shortcut connection input to Transfr block
        x = self.norm1(x)
        x = self.att.forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x # short conn for feed forward blck
        x = self.norm2(x)
        x = self.ff.forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
t_block = TransformerBlock(gpt_config.GPT_CONGIF_124M)
response = t_block(x)

print("INPUT SHAPE: ", x.shape, "\n")
print("OUTPUT shape: ", output.shape, "\n\n")

class GPTModel(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        self.token_emb = nn.Embedding(gpt_config["vocab_size"], gpt_config["emb_dim"])
        self.pos_emb = nn.Embedding(gpt_config["context_length"], gpt_config["emb_dim"])
        self.drop_emb = nn.Dropout(gpt_config["drop_rate"])
        self.trans_blck = nn.Sequential(
            *[TransformerBlock(gpt_config) for _ in range(gpt_config["n_layers"])]
        )
        self.final_norm = LayerNorm(gpt_config["emb_dim"])
        self.out_head = nn.Linear(gpt_config["emb_dim"], gpt_config["vocab_size"], bias=False)

    def forward(self, x):
        batch_sz, seq_len = x.shape
        token_embs = self.token_emb(x)

        pos_embs = self.pos_emb(
            torch.arange(seq_len, device=x.device)
        )

        x = token_embs + pos_embs
        x = self.drop_emb(x)
        x = self.trans_blck(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(gpt_config.GPT_CONGIF_124M)

output = model(batch)

print("INPUT BATCH SHAPE: ", batch.shape, "\n")
print("OUTPUT shape: ", output.shape)

total_params = sum(parm.numel() for parm in model.parameters())
print("Total params: ", total_params)

print("TOKEN EMB LAYER SHAPE: ", model.token_emb.weight.shape, "\n")
print("OUTPUT LAYER SHAPE: ", model.out_head.weight.shape)

total_gpt2_params = total_params - sum(prm.numel() for prm in model.out_head.parameters())
print("TOTAL GPT PARMAS: ", total_gpt2_params)

# TOTAL SIZE
total_size_in_bytes = total_params * 4 # (each param float32 ~ 4 bytes)
total_size_mb = total_size_in_bytes / (1024 * 1024)
print("MODEL SIZE: ", total_size_mb, " MB")

def generate_model_text(model, x, max_new_tokens, context_ln):
    for _ in range(max_new_tokens):
        x_size = x[:, -context_ln:]
        with torch.no_grad():
            logits = model(x_size)

            logits = logits[:, -1, :]
            prob = torch.softmax(logits, dim=-1)
            next_x = torch.argmax(prob, dim=-1, keepdim=True)
            x = torch.cat((x, next_x), dim=1)

    return x

tokenizer = tiktoken.get_encoding("gpt2")
initial_batch = "Hello, I am"
encoded = tokenizer.encode(initial_batch)
print("ENCODED: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("ENCODED TENSOR SHAPE: ", encoded_tensor.shape)

model.eval()
output = generate_model_text(model=model, 
                             x=encoded_tensor, 
                             max_new_tokens=6, 
                             context_ln=gpt_config.GPT_CONGIF_124M["context_length"])

print("OUTPUT: ", output)
print("OUTPUT len: ", len(output[0]))

decoded_text = tokenizer.decode(output.squeeze(0).tolist())
print("DECODED TXT: ", decoded_text)