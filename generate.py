from math import log
import torch
import torch.nn as nn
from .GPTModel import GPTModel
import tiktoken


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    args:
        idx: (batch_size, num_tokens) tensor in the current context
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size.
        # e.g. if LLM supports only 5 tokens, and the context size is 10,
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time stamp, so that
        # (batch_size, num_tokens, vocab_size) becomes (batch_size, vocab_size)
        logits = logits[:, -1, :]
        # probabilities has shape: (batch_size, vocab_size)
        probs = torch.softmax(logits, dim = -1)
        # idx_next has shape: (batch_size, 1)
        idx_next = torch.argmax(probs, dim = -1, keepdim = True)
        # Append sampled index to the running sequence, where idx has shape (batch_size, num_tokens + 1)
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    # Add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # remove batch dimension
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature = 1.0, top_k = None, eos_id = None):
    """"
    Generate text using temperature scaling and top_k method.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # if top_k is given, use it
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition = logits < min_val,
                input = torch.tensor(float('-inf')).to(logits.device),
                other = logits
            )

        # if temperature is given, use it
        if temperature is not None and temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)
    
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model,
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))

    model.train()