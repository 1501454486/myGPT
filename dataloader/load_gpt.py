import torch
import numpy as np



def assign(left, right):
    """
    Checks whether two tensors or arrays (left and right) have the same dimensions or shape and returns the right tensor as trainable PyTorch parameters.

    Args:
        left (torch.tensor):
        right (torch.tensor):

    Raises:
        ValueError:

    Returns:
        torch.nn.Parameter:
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):  #B
        q_w, k_w, v_w = np.split(  #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformerBlocks[b].attn.W_query.weight = assign(
            gpt.transformerBlocks[b].attn.W_query.weight, q_w.T)
        gpt.transformerBlocks[b].attn.W_key.weight = assign(
            gpt.transformerBlocks[b].attn.W_key.weight, k_w.T)
        gpt.transformerBlocks[b].attn.W_value.weight = assign(
            gpt.transformerBlocks[b].attn.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformerBlocks[b].attn.W_query.bias = assign(
            gpt.transformerBlocks[b].attn.W_query.bias, q_b)
        gpt.transformerBlocks[b].attn.W_key.bias = assign(
            gpt.transformerBlocks[b].attn.W_key.bias, k_b)
        gpt.transformerBlocks[b].attn.W_value.bias = assign(
            gpt.transformerBlocks[b].attn.W_value.bias, v_b)
        
        gpt.transformerBlocks[b].attn.out_proj.weight = assign(
            gpt.transformerBlocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformerBlocks[b].attn.out_proj.bias = assign(
            gpt.transformerBlocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.transformerBlocks[b].ff.layers[0].weight = assign(
            gpt.transformerBlocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformerBlocks[b].ff.layers[0].bias = assign(
            gpt.transformerBlocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformerBlocks[b].ff.layers[2].weight = assign(
            gpt.transformerBlocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformerBlocks[b].ff.layers[2].bias = assign(
            gpt.transformerBlocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.transformerBlocks[b].norm1.scale = assign(
            gpt.transformerBlocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformerBlocks[b].norm1.shift = assign(
            gpt.transformerBlocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformerBlocks[b].norm2.scale = assign(
            gpt.transformerBlocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformerBlocks[b].norm2.shift = assign(
            gpt.transformerBlocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
