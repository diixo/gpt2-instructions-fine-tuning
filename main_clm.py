import json
import torch
from torch.utils.data import DataLoader
from instruction_dataset import InstructionDataset, format_input
import tiktoken
from explanation import custom_collate_fn
from functools import partial
from utils import calc_loss_loader, train_model_simple, plot_losses
import time
from transformers import GPT2LMHeadModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 128,  # Context length
    "max_context_length": 1024,
}
num_workers = 0
batch_size = max(1, int(BASE_CONFIG["max_context_length"] // BASE_CONFIG["context_length"]))
num_epochs = 50
learning_rate = 5e-5
EOS_TOKEN_ID = 50256


tokenizer = tiktoken.get_encoding("gpt2")
file_path = "dataset.jsonl"


def extract_coreferenced_tokens(prompt, model, enc, max_new_tokens=20, temperature=0.0):
    input_ids = torch.tensor([enc.encode(prompt)]).to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=EOS_TOKEN_ID
    )

    # Берём только новые токены после prompt
    answer_ids = outputs[0].tolist()[input_ids.shape[1]:]
    answer = enc.decode(answer_ids).strip()
    return answer


if __name__ == "__main__":

    customized_collate_fn = partial(
        custom_collate_fn,
        allowed_max_length=BASE_CONFIG["context_length"],
        device=device,
    )


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
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(train_data, tokenizer)    # val_data
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )


    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=1)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=1)

    print(f"Initial training loss: {train_loss:.4f}")
    print(f"Initial validation loss: {val_loss:.4f}")


    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=None,    #val_data
        tokenizer=tokenizer
    )

    end_time = time.time()

    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    #############################################################################

    prompts = [
        { "instruction": "Metaphor of", "input": "\"very blue\"" },
        { "instruction": "Spell the word", "input": "\"Ocassion\"" },
        { "instruction": "Enumerate forms of word", "input": "\"run\"" },
        ]

    for item in prompts:
        prompt = format_input(item)
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 16,
            do_sample=False,                    # greedy decoding
            pad_token_id=EOS_TOKEN_ID
        )

        space_token_id = tokenizer.encode(" ")[0]
        generated_ids = outputs[0].tolist()

        if generated_ids[-1] == EOS_TOKEN_ID:
            generated_ids[-1] = space_token_id

        # Get only new tokens as answer:
        answer_txt = tokenizer.decode(generated_ids[input_ids.shape[1]:])
        print(32*"#" + f"\n{prompt}\n{answer_txt.strip()}")
