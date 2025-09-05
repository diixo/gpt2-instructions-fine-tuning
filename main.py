import json
import torch
from torch.utils.data import DataLoader
from instruction_dataset import InstructionDataset, format_input
import tiktoken
from explanation import custom_collate_fn
from functools import partial
from utils import generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model_simple, plot_losses
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
}
num_workers = 0
batch_size = 8
num_epochs = 5


customized_collate_fn = partial(
    custom_collate_fn,
    allowed_max_length=BASE_CONFIG["context_length"],
    device=device,
)

tokenizer = tiktoken.get_encoding("gpt2")
#print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


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
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(train_data, tokenizer)     # val_data
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
    )

    test_dataset = InstructionDataset(train_data, tokenizer)    # test_data
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print("sz:", inputs.shape[0], targets.shape[0])


    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Hello, I'm a language model,", tokenizer).to(device),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    print(token_ids_to_text(token_ids, tokenizer))

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print(f"Initial training loss: {train_loss:.4f}")
    print(f"Initial validation loss: {val_loss:.4f}")


    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

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
    # Test results:
    for entry in train_data[:3]:

        input_text = format_input(entry)

        token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"],
                eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
                generated_text[len(input_text):]
                .replace("### Response:", "")
                .strip()
        )
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-" * 20)

    #model_save_path = '/kaggle/working/gpt2-medium355M-sft.pth'
    #torch.save(model.state_dict(), model_save_path)