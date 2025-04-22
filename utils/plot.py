import matplotlib.pyplot as plt
import torch
from .temperature import softmax_with_temperature

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epochs_seen, train_losses, label = "Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = "Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc = "upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha = 0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def plot_temperature(temperatures, next_token_logits, vocab):
    scaled_probs = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize = (5, 3))
    for i, T in enumerate(temperatures):
        print("T: ", T)
        rects = ax.bar(x + i * bar_width, scaled_probs[i],
                    bar_width, label = f'Temperature = {T}')
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation = 90)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_values(epoch_seen, examples_seen, train_values, val_values, label = "loss"):
    fig, ax1 = plt.subplots(figsize = (5, 3))
    ax1.plot(epoch_seen, train_values, label = f"Training {label}")
    ax1.plot(epoch_seen, val_values, linestyle = "-.", label = f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha = 0)

    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()