import torch
import torch.nn as nn

from myGPT import dataloader
from myGPT.generate import generate_text_simple, text_to_token_ids, token_ids_to_text, generate_and_print_sample
from .utils.loss import cal_loss_batch, cal_loss_batch_classification
from .utils.loss import cal_loss_loader_classification
from .utils.loss import cal_loss_loader
from .dataloader.dataset import cal_acc_loader


def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    batches_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        # create iter for training
        train_iter = iter(train_loader)
        batch_count = 0

        # make sure that each epoch only handles batches given
        while batch_count < batches_per_epoch:
            try:
                input_batch, target_batch = next(train_iter)
            except StopIteration:
                # if iteration ran out, create a new.
                train_iter = iter(train_loader)
                input_batch, target_batch = next(train_iter)

            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            batch_count += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Calculates the loss over the training and validation set while ensuring the model is in evaluation mode with gradient tracking and dropout disabled, when calculating loss over the training and validation sets
    """
    # dropout is disabled during evaluation for stable, reproducible results
    model.eval()
    # disable gradient tracking, which is not required during evaluation, to reduce the computational overhead
    with torch.no_grad():
        train_loss = cal_loss_loader(
            data_loader = train_loader,
            model = model,
            device = device,
            num_batches = eval_iter
        )
        val_loss = cal_loss_loader(
            data_loader = val_loader,
            model = model,
            device = device,
            num_batches = eval_iter
        )
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter
):
    """
    Train the model for classification task, deciding whether the message to be spam or ham.
    Almost the same as train_model_simple.

    Args:
        model
        train_loader
        val_loader
        optimizer
        device
        num_epochs
        eval_freq
        eval_iter

    Returns:
        train_losses: List of losses during training
        val_losses: List of losses during validation
        train_accs: List of accuracies during training
        val_accs: List of accuracies during validation
        examples_seen: number of examples during training
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    batches_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        # create iter for training
        train_iter = iter(train_loader)
        batch_count = 0

        # make sure that each epoch only handles batches given
        while batch_count < batches_per_epoch:
            try:
                input_batch, target_batch = next(train_iter)
            except StopIteration:
                # if iteration ran out, create a new.
                train_iter = iter(train_loader)
                input_batch, target_batch = next(train_iter)

            optimizer.zero_grad()
            loss = cal_loss_batch_classification(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1
            batch_count += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classification(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
        train_acc = cal_acc_loader(
            train_loader, model, device, num_batches = eval_iter
        )
        val_acc = cal_acc_loader(
            val_loader, model, device, num_batches = eval_iter
        )

        print(f"Training accuracy: {train_acc * 100 :.2f}% | ", end = "")
        print(f"Validation accuracy: {val_acc * 100 :.2f}%")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model_classification(model, train_loader, val_loader, device, eval_iter):
    """
    Almost the same as evaluate_model, but adjusted for classification.
    """
    # dropout is disabled during evaluation for stable, reproducible results
    model.eval()
    # disable gradient tracking, which is not required during evaluation, to reduce the computational overhead
    with torch.no_grad():
        train_loss = cal_loss_loader_classification(
            data_loader = train_loader,
            model = model,
            device = device,
            num_batches = eval_iter
        )
        val_loss = cal_loss_loader_classification(
            data_loader = val_loader,
            model = model,
            device = device,
            num_batches = eval_iter
        )
    model.train()
    return train_loss, val_loss