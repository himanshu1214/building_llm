import numpy as np
import gpt_config
import torch
from gpt_model import GPTModel
import gpt_config
from model_configs import model_configs_map
from utils import text_to_token, token_ids_to_text
from pretraining import generature_with_decoding_strat
import tiktoken

## LOADING PRETRAINED GPT-2 model
from gpt_download import download_and_load_gpt2


def assign(name, param, array):
    """
    Helper method for debugging load weight
    Identifying the current shape of matrices and matrices from gpt model
    Raise error
    """
    if param.shape != array.shape:
        print(f"Shape mismatch: {name}", param.shape, array.shape)
        raise ValueError("Shape Mismatch Stop loading ")

    return torch.nn.Parameter(torch.tensor(array, dtype=param.dtype))

def load_weight(gpt, params):
    gpt.pos_emb.weight = assign("pos-emb-wt", gpt.pos_emb.weight, params['wpe'])
    gpt.token_emb.weight = assign("token-emb-wt",gpt.token_emb.weight, params['wte'])

    for ind, blk in enumerate(params['blocks']): # total 12 block

        ##WEIGHT
        q, k, v = np.split(blk["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trans_blck[ind].att.W_query.weight = assign("q_wt",
            gpt.trans_blck[ind].att.W_query.weight, q.T
        )
        gpt.trans_blck[ind].att.W_key.weight = assign("k_wt",
            gpt.trans_blck[ind].att.W_key.weight, k.T
        )
        gpt.trans_blck[ind].att.W_value.weight = assign("v_wt",
            gpt.trans_blck[ind].att.W_value.weight, v.T
        )


        ##BIAS
        q_b, k_b, v_b = np.split(blk["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trans_blck[ind].att.W_query.bias = assign("q_bias",
            gpt.trans_blck[ind].att.W_query.bias, q_b
        )
        gpt.trans_blck[ind].att.W_key.bias = assign("k_bias",
            gpt.trans_blck[ind].att.W_key.bias, k_b
        )
        gpt.trans_blck[ind].att.W_value.bias = assign("v_bias",
            gpt.trans_blck[ind].att.W_value.bias, v_b
        )

        # OUTPUT weight
        gpt.trans_blck[ind].att.output_dims.weight = assign("output_wt",
            gpt.trans_blck[ind].att.output_dims.weight, 
            blk["attn"]["c_proj"]["w"].T
        )

        # OUTPUT bias
        gpt.trans_blck[ind].att.output_dims.bias = assign("output_bias",
            gpt.trans_blck[ind].att.output_dims.bias, 
            blk["attn"]["c_proj"]["b"]
        )

        # FFeedForwardeedForward-0 Layer Weight
        gpt.trans_blck[ind].ff.layers[0].weight = assign("ff-0-wt",
            gpt.trans_blck[ind].ff.layers[0].weight,
            blk["mlp"]["c_fc"]["w"].T
        )
        # FeedForward-0 Layer Bias
        gpt.trans_blck[ind].ff.layers[0].bias = assign("ff-0-bias",
            gpt.trans_blck[ind].ff.layers[0].bias,
            blk["mlp"]["c_fc"]["b"]
        )
        # FeedForward-2 Layer Weight
        gpt.trans_blck[ind].ff.layers[2].weight = assign("ff-2-wt",
            gpt.trans_blck[ind].ff.layers[2].weight,
            blk["mlp"]["c_proj"]["w"].T
        )
        # FeedForward-2 Layer Bias
        gpt.trans_blck[ind].ff.layers[2].bias = assign("ff-2-bias",
            gpt.trans_blck[ind].ff.layers[2].bias,
            blk["mlp"]["c_proj"]["b"]
        )

        # Norm1 Scale
        gpt.trans_blck[ind].norm1.scale = assign("norm-1-scale",
            gpt.trans_blck[ind].norm1.scale,
            blk["ln_1"]["g"])
        # Norm1 Shift
        gpt.trans_blck[ind].norm1.shift = assign("norm-1-shift",
            gpt.trans_blck[ind].norm1.shift,
            blk["ln_1"]["b"])
        # Norm2 Scale
        gpt.trans_blck[ind].norm2.scale = assign("norm-2-scale",
            gpt.trans_blck[ind].norm1.scale,
            blk["ln_2"]["g"])   
        # Norm2 Shift
        gpt.trans_blck[ind].norm2.shift = assign("norm-2-shift",
            gpt.trans_blck[ind].norm2.shift,
            blk["ln_2"]["b"])
        

    gpt.final_norm.scale = assign("final-norm-scale", gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign("final-norm-shift", gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign("out-head-weight", gpt.out_head.weight, params["wte"])

if __name__ == '__main__':

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    print("GPT2-settings: ", settings) # similar to GPT CONFIG 124M
    print("GPT2-model-weight-keys: ", params.keys())
    print("GPT2-model weights: ", params["wte"]) # model weights
    print("GPT2-model-weights-shape: ", params["wte"].shape)
    #
    mode_name_select = "gpt2-small-124M"

    new_config = gpt_config.GPT_CONGIF_124M.copy()
    new_config.update(model_configs_map[mode_name_select])
    new_config.update({"qkv_bias": True})
    gpt = GPTModel(new_config)
    gpt.eval()
    device = torch.device("cuda")
    tokenizer = tiktoken.get_encoding("gpt2")

    print(new_config)
    load_weight(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        token_ids = generature_with_decoding_strat(
        gpt, 
        batch_emb=text_to_token("every effort moves you", tokenizer).to(device), 
        max_new_tokens=25, 
        context_ln=gpt_config.GPT_CONGIF_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("TEXT GENERATED USING GPT2 model: ", token_ids_to_text(token_ids=token_ids, tokenize=tokenizer)) 