import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetv1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire txt
        token_ids = tokenizer.encode(txt)

        # using a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    


def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 4, persistent_workers = True):
    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # create a dataset
    dataset = GPTDatasetv1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers,
        persistent_workers = persistent_workers
    )

    return dataloader


def cal_acc_loader(data_loader, model, device, num_batches = None):
    """
    determine the classification accuracy

    Args:
        data_loader (DataLoader)
        model
        device
        num_batches: Defaults to None.

    Returns:
        float: accuracy
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            
            predicted_labels = torch.argmax(logits, dim = -1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    
    return correct_predictions / num_examples