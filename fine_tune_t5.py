import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch
# Change cache directory to another drive with more space
os.environ['HF_HOME'] = 'D:/huggingface'  # Ensure this path exists on your system

# Load the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
# Load the PubMed summarization dataset with trust_remote_code set to True
# Load only a small subset of the dataset for testing
ds_document = load_dataset("ccdv/pubmed-summarization", "document", split='train[:1%]', trust_remote_code=True)
ds_section = load_dataset("ccdv/pubmed-summarization", "section", split='train[:1%]', trust_remote_code=True)

# Preprocess function for tokenizing input and target text
def preprocess_function(examples):
    inputs = [ex for ex in examples['article']]
    targets = [ex for ex in examples['abstract']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the dataset
tokenized_ds_document = ds_document.map(preprocess_function, batched=True)
tokenized_ds_section = ds_section.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize the Trainer for the document dataset
trainer_document = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_document,
    eval_dataset=tokenized_ds_document
)

# Fine-tune the model on the document dataset
trainer_document.train()

# Save the model for the document dataset
model.save_pretrained('./models/fine_tuned_t5_document')
tokenizer.save_pretrained('./models/fine_tuned_t5_document')

# Initialize the Trainer for the section dataset
trainer_section = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_section,
    eval_dataset=tokenized_ds_section
)

# Fine-tune the model on the section dataset
trainer_section.train()

# Save the model for the section dataset
model.save_pretrained('./models/fine_tuned_t5_section')
tokenizer.save_pretrained('./models/fine_tuned_t5_section')
