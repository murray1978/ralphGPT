import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        """
        Initializes the dataset with tokenized data and the block size for sequences.
        :param data: The tokenized data (e.g., train_data or val_data)
        :param block_size: The length of each sequence block (context length)
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Number of samples in the dataset
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Get a single example from the dataset at index idx
        """
        # Get input x (block_size tokens) and target y (next token after x)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
