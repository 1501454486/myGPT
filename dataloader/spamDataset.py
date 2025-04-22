import torch
from torch.utils.data import Dataset
import pandas as pd


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        """
        args:
            csv_file: path of dataset file
            tokenizer
            max_length
            pad_token_id: used for padding, 50256(<|endoftext|> by default)
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenized texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they're longer than max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        # Pad sequences to the longest sequence
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        """
        given  index, get the corresponding item

        Args:
            index (int): index we want

        Returns:
            tuple:
                - torch.Tensor: encoded text tensor of shape (sequence_length,) with dtype torch.long
                - torch.Tensor: label tensor (scalar) with dtype torch.long
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype = torch.long),
            torch.tensor(label, dtype = torch.long)
        )
    
    def __len__(self):
        """
        get the length of this dataset

        Returns:
            int: length of dataset
        """
        return len(self.data)
    
    def _longest_encoded_length(self):
        """
        Return the longest encoded length of this dataset.

        returns:
            int: max length of this datset
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length

        return max_length