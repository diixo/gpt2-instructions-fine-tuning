import json
import torch
from torch.utils.data import Dataset, DataLoader
from instruction_dataset import InstructionDataset
import tiktoken
from explanation_1 import custom_collate_fn
from functools import partial


customized_collate_fn = partial(
    custom_collate_fn,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    allowed_max_length=1024
)

tokenizer = tiktoken.get_encoding("gpt2")
#print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


batch_size = 8


if __name__ == "__main__":

    file_path = "dataset.jsonl"
    train_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))


    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    val_dataset = InstructionDataset(train_data, tokenizer)     # val_data
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=0
    )

    test_dataset = InstructionDataset(train_data, tokenizer)    # test_data
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=0
    )
