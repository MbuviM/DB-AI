from transformers import pipeline, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("text", data_files="data/diabetes.txt")

# Load the pipeline
pipe = pipeline("text2text-generation")

checkpoint="facebook/mbart-large-50-many-to-one-mmt"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization
def tokenize_function(Words):
    return pipe.tokenizer(Words["text"], truncation=True)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Data Loaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator
)

# Inspect Batch
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)