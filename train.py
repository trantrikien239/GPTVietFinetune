import argparse
import yaml

import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

import wandb
import huggingface_hub



# Parse training arguments
parser = argparse.ArgumentParser(description='Train a model on a dataset')
# parser.add_argument('--base_model', type=str, help='the huggingface model to use as a base')
# parser.add_argument('--dataset', type=str, default='data/clean/health_qa_train.parquet', help='the dataset to use for fine-tuning')
parser.add_argument('--config', type=str, default='configs/light.yaml', help='the config file to use for training')
parser.add_argument('--wandb_key', type=str, help='key for wandb login')
parser.add_argument('--huggingface_key', type=str, help='key for huggingface login')
parser.add_argument('--output_dir', type=str, default='./output', help='the directory to save the model to')
# parser.add_argument('--max_length', type=int, default=512, help='the maximum length of the input sequence')

args = parser.parse_args()

# Login to wandb and huggingface
if args.wandb_key is not None:
    print("====== Logging into wandb ======")
    wandb.login(key=args.wandb_key)
if args.huggingface_key is not None:
    print("====== Logging into huggingface ======")
    huggingface_hub.login(args.huggingface_key)


# Load training arguments from the config file, config is a yaml file
if args.config is not None:
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise ValueError("No config file specified")

BASE_MODEL_PATH = config['base_model']
DATASET_PATH = config['dataset']
MAX_LENGTH = config['max_length']

# Load the dataset
print("====== Loading dataset ======")
dataset = load_dataset('parquet', data_files=DATASET_PATH, split='train')


# Load the model and tokenizer
print("====== Loading model and tokenizer ======")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True,
    truncation=True, max_length=MAX_LENGTH)

# Load configs for training
print("====== Loading configs ======")
peft_config = LoraConfig(**config['lora'])

training_arguments = TrainingArguments(
    output_dir=args.output_dir,
    fp16=True,
    group_by_length=False,
    **config['training']
)

# Create the trainer
print("====== Creating trainer ======")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Move normalization layers to float32 for training stability
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train the model
wandb.init(project="GPTVietFinetune", config=config)
print("====== Training ======")
trainer.train()