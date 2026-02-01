
#requirement.txt

pip install torch transformers datasets peft PyPDF2 langchain faiss-cpu

# ------------ PDF -> Text Dataset Creation ------------

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from datasets import Dataset

path = "D:\\Freelancer\\drive-download-20240913T032445Z-001\\physicsmaterial.pdf"

pdfreader = PdfReader(path)
raw_text = ""

for page in pdfreader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + "\n"

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = splitter.split_text(raw_text)
print(f"Created {len(texts)} chunks")

# Build a simple dataset — we'll train the model to answer generic questions
examples = []
for t in texts:
    examples.append({"text": t})

dataset = Dataset.from_list(examples)

#Now dataset holds our knowledge base text ready for training.

#Tokenize dataset
from transformers import AutoTokenizer

model_name = "gpt2"  # example local model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = dataset.map(tokenize_fn, batched=True)

#Lora config, rank = 16 is used to decompose the delta weight dimention (m*n) matrix into simplified ones (m*r & r*n)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # selective attention layers
    lora_dropout=0.05,
)

#load model with lora config definition. Now only the target_modules layers are suseptible to fine tune parameters, rest other layers will have the old parameters only.
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)

#  ↓ This wraps the model to freeze base weights and add A/B LoRA adapters
model = get_peft_model(model, lora_config)

print("Trainable params:")
model.print_trainable_parameters()

#Train with fine tuned arguments
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=100,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

#Inference after fine tuning
prompt = "Explain the truth table for logic gates."

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

