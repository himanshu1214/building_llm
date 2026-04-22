import torch
import torch.nn as nn

from tokenization import create_data_loader_v1

vocab_size = 50257
output_dim = 3
txt1 = "Your Journey starts with one step again"
new_data_loader = create_data_loader_v1(
    txt1, batch_size=1, max_length=6, stride=1, shuffle=False
)
iter_new_dt = iter(new_data_loader)
inputs, target = next(iter_new_dt)


torch.manual_seed(123)
# each token is represent in 3Dimensional space
# in GPT-2 uses emb_dim ~ 768
ex_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

ex_token_embedding = ex_embedding_layer(inputs[0])  # shape (6, 3)

print("INPUT TOKEN SHAPE: ", inputs.shape)
print("INPUT_TOKEN: ", inputs)

print("TOKEN EMBEDDINGS :", ex_token_embedding)
print("TOKEN EMBEDDINGS SHAPE: ", ex_token_embedding.shape)
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
attn_scores = torch.empty(6, 6)
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


################################################################
# Implementation of GPT using trainable weights
d_in = ex_token_embedding.shape[1]  # ~3
d_out = 2  #
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_val = torch.nn.Parameter(
    torch.rand(d_in, d_out), requires_grad=False
)  # make requires_grad = True when model train for wts updation


query2 = ex_token_embedding[1] @ W_query
key2 = ex_token_embedding[1] @ W_key
value2 = ex_token_embedding[1] @ W_val

print("QUERY2", query2)

query = ex_token_embedding @ W_query
keys = ex_token_embedding @ W_key
value = ex_token_embedding @ W_val

print("key shape", keys.shape)
print("value shape", keys.shape)

# Attention score of w22
key_2 = keys[1]
attn_score_22 = query2 @ key_2

attn_score_2 = query2 @ keys.T
print("Attn scores for 2nd token", attn_score_2)

d_k = keys.shape[-1]
print("D_K", d_k)

attn_wts_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
print("atten_normalized_wts", attn_wts_2)

context_vector_2 = attn_wts_2 @ value
print("CONTEXT VECTOR 2", context_vector_2)


class SelfAttentionV1(torch.nn.Module):

    def __init__(self, d_in, d_out):
        """
        Initialize keys, values, queries matrix for each token embedding input
        """
        super().__init__()

        # initialize the queries , val, key
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_val = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.attn_wts = None
        self.attn_score = None
        self.d_k = None
        self.val = None

    def feed_forward(self, token_embeddding):
        """
        this method is used to create context vectors for each token
        Input x: token tensor same
        """
        keys = token_embeddding @ self.W_key  # step 1 find key
        query = (
            token_embeddding @ self.W_query
        )  # step 1 dot  [m x n ] x [n x m ] ~ [m x m ]
        self.val = token_embeddding @ self.W_val

        self.attn_score = query @ keys.T

        self.d_k = keys.shape[-1]
        self.attn_wts = torch.softmax(self.attn_score / d_k**0.5, dim=-1)

        context_vec = self.attn_wts @ self.val
        return context_vec


d_in = 3
d_out = 2
slf_attn = SelfAttentionV1(d_in=d_in, d_out=d_out)

print("SLF_ATN : ", slf_attn.feed_forward(ex_token_embedding))

attn_score = slf_attn.attn_score
attn_wts = slf_attn.attn_wts
context_ln = attn_wts.shape[0]
mask_simple = torch.tril(torch.ones(context_ln, context_ln))
print("MASK", mask_simple)

# Element wise multiplication
masked_attn_wts = attn_wts * mask_simple

print("MASKED ATTN WTS", masked_attn_wts)

row_sums = masked_attn_wts.sum(dim=-1, keepdim=True)
masked_normalized_wts = masked_attn_wts / row_sums
print("MASKED NORMALIZED WTS : ", masked_normalized_wts)

## Using inf as mask
inf_mask = torch.triu(torch.ones(context_ln, context_ln), diagonal=1)
masked_inf_scores = attn_score.masked_fill(inf_mask.bool(), -torch.inf)
print("MASKED INF SCORES: ", masked_inf_scores)

attn_inf_wts = torch.softmax(masked_inf_scores / slf_attn.d_k**0.5, dim=1)
print("ATTN INF WTS: ", attn_inf_wts)

context_inf_vec = attn_inf_wts @ slf_attn.val
print("CONTEXT INF VEC: ", context_inf_vec)


# DROPOUT
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
one_tnsr = torch.ones(6, 6)  # matrix of 1s
print("DROPOUT TST: ", dropout(one_tnsr))


class CompactCausalAttention(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        context_ln,
        qkv_bias=False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)

        # useful during model training for keeping model params and tensors are
        # on same device
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_ln, context_ln), diagonal=1)
        )

    def feed_forward(self, token_embed):
        """this method is used to create context
        by masking using dropout

        """
        _, num_token, __ = token_embed.shape
        print("NUM TOKEN", num_token, "E", _, "EE", __)
        cc_keys = self.W_key(token_embed)  # this does the dot product
        cc_values = self.W_value(token_embed)
        cc_query = self.W_query(token_embed)
        cc_attn_scores = cc_query @ cc_keys.transpose(1, 2)

        # mask fill in-place with inf for future indexes
        cc_attn_scores.masked_fill_(
            self.mask.bool()[:num_token, :num_token], -torch.inf
        )

        cc_attn_wts = torch.softmax(cc_attn_scores / cc_keys.shape[1] ** 0.5, dim=-1)

        cc_attn_wts = self.dropout(cc_attn_wts)
        cc_context_vec = cc_attn_wts @ cc_values
        return cc_context_vec


# Support batch inputs
torch.manual_seed(123)
batch = torch.stack((ex_token_embedding, ex_token_embedding), dim=0)
print("BATCH SHAPE: ", batch.shape)
print("BATCH : ", batch)
context_ln = batch.shape[1]
ca = CompactCausalAttention(d_in=d_in, d_out=d_out, dropout=0, context_ln=context_ln)
cc_context_vec = ca.feed_forward(batch)
print("CC CONTEXT VEC SHAPE: ", cc_context_vec.shape)
print("CC CONTEXT VEC: ", cc_context_vec)


class MultiHeadAttentionWrapper(torch.nn.Module):
    """
    This class implements Causal class and generating multiple heads and concating it.
    This is done in sequential manner
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                CompactCausalAttention(
                    d_in=d_in,
                    d_out=d_out,
                    dropout=dropout,
                    context_ln=context_length,
                    qkv_bias=qkv_bias,
                )
                for _ in num_heads
            ]
        )

    def feed_forward(self, token_embed):
        return torch.cat([head(token_embed) for head in self.heads], dim=-1)


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_ln, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # ex: 512 Dim, then 512 // 8 ~ 64 Dim each head, total 8 heads
        self.dropout = torch.nn.Dropout(dropout)
        self.d_out = d_out
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_ln, context_ln), diagonal=1)
        )
        self.output_dims = torch.nn.Linear(d_out, d_out)

    def forward(self, token_embed):
        """
        Using parallel processing to get multi heads implementation
        this optimizes MultiHeadAttentionWrapper

        (2, 4, 8)          # input
            ↓
        (2, 4, 2, 4)      # split heads (.view)
            ↓
        (2, 2, 4, 4)      # transpose
            ↓
        (2, 2, 4, 4)      # attention
            ↓
        (2, 4, 2, 4)      # transpose back
            ↓
        (2, 4, 8)         # merge heads (.view)
        """

        # print("TOKEN EMBED SHAPE: ", token_embed.shape, "\n")
        b, num_token, d_in = token_embed.shape
        mm_keys = self.W_key(token_embed)
        mm_values = self.W_value(token_embed)
        mm_queries = self.W_query(token_embed)

        # print("MM - QUERY SHAPE: ", mm_queries.shape, "\n")

        mm_keys = mm_keys.view(b, num_token, self.heads, self.head_dim)
        mm_values = mm_values.view(b, num_token, self.heads, self.head_dim)
        mm_queries = mm_queries.view(b, num_token, self.heads, self.head_dim)

        # print("MM VIEW- QUERY SHAPE:  ", mm_queries.shape, "\n")
        # batch, tokens, heads, head_dim
        # ex: (2, 4, 8) --> (2, 4, 2, 4), so => 8 --> (2 head, 4 head_dim)
        mm_keys = mm_keys.transpose(1, 2)
        mm_values = mm_values.transpose(1, 2)
        mm_queries = mm_queries.transpose(1, 2)

        # print("MM TRANSPOSE- QUERY SHAPE: ", mm_queries.shape, "\n")
        # ex: (2, 4, 2, 4) --> (2, 2, 4, 4)
        # batch, heads, tokens, head_dim
        mm_attn_scores = mm_queries @ mm_keys.transpose(2, 3)
        mm_mask_bool = self.mask.bool()[:num_token, :num_token]

        mm_attn_scores.masked_fill_(mm_mask_bool, -torch.inf)

        mm_attn_wts = torch.softmax(mm_attn_scores / mm_keys.shape[-1] ** 0.5, dim=-1)
        mm_attn_wts = self.dropout(mm_attn_wts)

        # Transpose (2, 4, 2, 4)
        # (batch, tokens, heads, head_dim)
        mm_context_vec = (mm_attn_wts @ mm_values).transpose(1, 2)

        # merge heads ~ (2, 4, 8)
        mm_context_vec = mm_context_vec.contiguous().view(b, num_token, self.d_out)

        mm_context_vec = self.output_dims(mm_context_vec)
        return mm_context_vec


torch.manual_seed(123)
# num_token ~ number of token in each batch
# d_out ~ embedding dimension
# head_dim ~ number of dimension for each head


batch_size, context_ln, d_in = batch.shape
d_out = 2

mha = MultiHeadAttention(d_in, d_out, context_ln, 0, num_heads=2)
mm_context_vec = mha.forward(batch)
print("MULTI HEAD ATTENTION VEC SHAPE: ", mm_context_vec.shape)
print("MULTI HEAD ATTENTION: ", mm_context_vec)
