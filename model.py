# Import necessary libraries
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import tensorflow as tf

# Function to load the local dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        text_data = file.readlines()
    return Dataset.from_dict({"text": text_data})

# Function for tokenization
def tokenization(data, tokenizer):
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors="pt")

        # Add labels for causal language modeling
    encoded_input["labels"] = encoded_input["input_ids"].clone()
    return encoded_input

# Function to preprocess the dataset
def preprocess_dataset(dataset, tokenizer):
    return dataset.map(lambda x: tokenization(x["text"], tokenizer), batched=True)

# Main function
def main():
    # Load model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set the pad_token to be the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    file_path = "data/diabetes.txt"
    diabetes_data = load_dataset(file_path)
    
    # Access the 21st element of the dataset (0-indexed, so it's the 20th index)
    sample_text = diabetes_data["text"][20]
    print(f"Sample text: {sample_text}")

    # Tokenize the entire dataset
    tokenized_data = preprocess_dataset(diabetes_data, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")

if __name__ == "__main__":
    main()
