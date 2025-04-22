import torch
import torch.nn.functional as F


def cal_loss_batch(input_batch, target_batch, model, device):
    # the transfer to a given device allows us to transfer the data to a GPU
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def cal_loss_batch_classification(input_batch, target_batch, model, device):
    """
    Calculate a single batch loss in classification task

    Args:
        input_batch (Tensor): _description_
        target_batch (Tensor): _description_
        model
        device

    Returns:
        float: loss
    """
    # Before we cal the loss, move them to the same device first
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 1. cal logits, note that we only care about the last row of elements
    logits = model(input_batch)[:, -1, :]
    # 2. use cross entropy
    loss = F.cross_entropy(logits, target_batch)
    return loss



def cal_loss_loader_classification(data_loader, model, device, num_batches = None):
    """
    use `cal_loss_batch_classification` to compute loss for all batches in a data loader

    Args:
        data_loader
        model
        device
        num_batches

    Returns:
        float: loss over all batches of a data_loader
    """
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch_classification(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches