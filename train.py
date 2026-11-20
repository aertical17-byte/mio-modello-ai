"""Addestra un piccolo modello GPT da zero. Eseguito nel cloud (GitHub Actions)."""
import os
import math
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "mio-modello-ai")
DATASET = os.environ.get("DATASET", "roneneldan/TinyStories")
N_TRAIN = int(os.environ.get("N_TRAIN", "1500"))
N_VAL = int(os.environ.get("N_VAL", "200"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "128"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))
N_LAYER = int(os.environ.get("N_LAYER", "2"))
N_HEAD = int(os.environ.get("N_HEAD", "2"))
N_EMBD = int(os.environ.get("N_EMBD", "128"))

print("=== Config ===")
print(f"model={MODEL_NAME} dataset={DATASET} train={N_TRAIN} val={N_VAL}")
print(f"block={BLOCK_SIZE} batch={BATCH_SIZE} epochs={EPOCHS} lr={LEARNING_RATE}")
print(f"layers={N_LAYER} heads={N_HEAD} embd={N_EMBD}")

print("=== Caricamento dataset ===")
ds = load_dataset(DATASET)
train_raw = ds["train"].select(range(N_TRAIN))
if "validation" in ds:
    val_raw = ds["validation"].select(range(N_VAL))
else:
    val_raw = ds["train"].select(range(N_TRAIN, N_TRAIN + N_VAL))

print("=== Tokenizer ===")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_train = train_raw.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_raw.map(tokenize_function, batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_val = tokenized_val.map(group_texts, batched=True, batch_size=1000)
print("Train blocks:", len(lm_train), "Val blocks:", len(lm_val))

print("=== Creazione modello ===")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_inner=N_EMBD * 4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parametri: {n_params/1e6:.2f} M")

print("=== Training ===")
training_args = TrainingArguments(
    output_dir=f"./{MODEL_NAME}",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    report_to="none",
    push_to_hub=False,
    use_cpu=not torch.cuda.is_available(),
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=data_collator,
)
trainer.train()

print("=== Valutazione ===")
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

print("=== Generazione di esempio ===")
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("=== Salvataggio ===")
save_path = f"./{MODEL_NAME}-final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("Modello salvato in:", save_path)
print("DONE")
