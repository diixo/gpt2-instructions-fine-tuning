import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        if len(token_ids) <= max_length:
            input_chunk = token_ids
            target_chunk = token_ids[1:] + [50256]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
        else:
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i: i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    txt = "Hello, I am language model!"

    ds = GPTDatasetV1(txt, tokenizer, 8, 1)
    print(ds.input_ids)
    print(ds.target_ids)
