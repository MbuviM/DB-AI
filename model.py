import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Load the dataset
df = pd.read_csv('data/prepared_data.csv')
dataset = Dataset.from_pandas(df)

# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the model and tokenizer
checkpoint = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization function
def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['response_text']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True,
    adafactor=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

