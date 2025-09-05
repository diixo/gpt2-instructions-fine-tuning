# gpt-instruction-ft


### 1. CLM-mode (like GPT-2 by default)

**Causal LM** - model learns to predict all text, including **Question**, **Input**, **Answer**, **EOS**.


### 2. SFT-mode (instructional fine-tune)

**Supervised Fine-tuning** — model learns only Answer, but **Question + Input** is masked (-100).


### How attention connects a question with an answer
```
[ List all forms of word: run ] --> внимание (context)
[ run, runs, ran, running<EOS> ] --> loss вычисляется только здесь
```

* На каждом шаге генерации токена ответа:

- 1. Модель смотрит на все предыдущие токены в inputs, включая замаскированный вопрос.

- 2. Attention позволяет модели “помнить”, что вопрос был "List all forms of word: run".

* Loss обновляет только те веса, которые нужны для предсказания ответа.
```
Inputs (all tokens) ----> Transformer layers ----> hidden states ----> Prediction next token
                   ↑
                Attention
                   |
Masked labels (-100) for question
                    ↓
          Loss calculated only on Question
```

* Маскировка вопроса не блокирует использование его как контекста.
* Модель учится: если вижу этот вопрос → выдаю этот ответ.
