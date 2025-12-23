import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
OUTPUT_DIR = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="train.jsonl")

def tokenize(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
