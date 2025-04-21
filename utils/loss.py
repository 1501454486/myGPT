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