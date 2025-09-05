
import tiktoken
import torch


def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def item_collate_fn(batch, tokenizer, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    input_ids_list, labels_list = [], []
    max_len = 0

    # Токенизация и маскирование
    for item in batch:
        question_ids = tokenizer.encode(item["question"])
        answer_ids = tokenizer.encode(item["answer"]) + [pad_token_id]
        # добавляю в конец принудительно endoftext, он не будет маскироваться на -100, потомучто является частью ответа

        input_ids = question_ids + answer_ids
        labels = [-100] * len(question_ids) + answer_ids  # маскируем вопрос

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        max_len = max(max_len, len(input_ids))

    if allowed_max_length is not None and allowed_max_length >= max_len:
        max_len = allowed_max_length
    else:
        # TODO:
        max_len = max_len


    # Padding
    padded_inputs = []
    padded_labels = []
    for inp, lbl in zip(input_ids_list, labels_list):
        padded_inputs.append(torch.tensor(inp + [pad_token_id] * (max_len - len(inp))))
        padded_labels.append(torch.tensor(lbl + [ignore_index] * (max_len - len(lbl))))

    inputs_tensor = torch.stack(padded_inputs).to(device)
    labels_tensor = torch.stack(padded_labels).to(device)

    return inputs_tensor, labels_tensor



def test_logit_neg_100():

    logits_1 = torch.tensor(
        [[-1.0, 1.0],
        [-0.5, 1.5]]
    )
    targets_1 = torch.tensor([0, 1])

    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)

    # -----------
    logits_2 = torch.tensor(
        [[-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5]]  # New 3rd training example
    )
    targets_2 = torch.tensor([0, 1, 1])

    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)

    # -----------
    targets_3 = torch.tensor([0, 1, -100])  # ignore 3rd row

    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)


########################################################################


if __name__ == "__main__":

    if False:
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8 , 9]

        batch = [inputs_1, inputs_2, inputs_3]


        inputs, targets = custom_collate_draft_2(batch)
        print(inputs)
        print(targets)

        print(24 * "*")
        inputs, targets = custom_collate_fn(batch)
        print(inputs)
        print(targets)

        #test_logit_neg_100()

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = [
        {"question": "Hello, Am I really language model", "answer": "yes"},
        {"question": "Hello, Am I language model", "answer": "yes"},
        ]
    inputs, labels = item_collate_fn(batch, tokenizer, allowed_max_length=None)
    print("inputs:", inputs)
    print("labels:", labels)
