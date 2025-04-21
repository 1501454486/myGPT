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