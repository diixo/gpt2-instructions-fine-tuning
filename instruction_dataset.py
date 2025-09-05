import json
import torch
from torch.utils.data import Dataset



def format_input(entry):
    instruction_text = (f"\n\n### Instruction:\n{entry['instruction']}")
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionItemDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        entry = self.data[idx]
        instruction = entry["instruction"]
        input_text = f" {entry['input']}" if entry['input'] else ""
        return {
            "question": instruction.strip() + input_text,
            "answer": entry["output"]
        }

    def __len__(self):
        return len(self.data)