import torch

def classify_review(text, model, tokenizer, device, max_length = None, pad_token_id = 50256):
    """
    Classifies a text review as spam or not spam using a pre-trained model.
    Follows data preprocessing steps similar to those used in the `SpamDataset`.

    Args:
        text (str): The review text to classify
        model (torch.nn.Module): The pre-trained classification model
        tokenizer: The tokenizer used to encode the text
        device (torch.device): The device to run inference on (CPU or GPU)
        max_length (int, optional): Maximum sequence length. Defaults to None.
        pad_token_id (int, optional): Token ID used for padding. Defaults to 50256.

    Returns:
        string: Classification result, either "spam" or "not spam"
    """
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # add batch dimension
    input_tensor = torch.tensor(input_ids, device = device).unsqueeze(0)

    # model inference without gradient tracking
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim = -1).item()

    return "spam" if predicted_label == 1 else "not spam"