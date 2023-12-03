import os
import argparse
import yaml

import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from tqdm.auto import tqdm
from utils import template_gen

parser = argparse.ArgumentParser(description='Arguments for inference')
parser.add_argument('--config', type=str, default='configs/light.yaml', help='the config file to use for training')
parser.add_argument('--adapter_path', type=str, help='the path to the pretrained adapter model')
parser.add_argument('--dataset', type=str, help='the dataset to inference on, parquet format')
parser.add_argument('--max_length', type=int, default=512, help='the maximum length of the input sequence')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for sampling')
parser.add_argument('--top_k', type=int, default=20, help='top_k for sampling')
parser.add_argument('--top_p', type=float, default=0.9, help='top_q for sampling')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load training arguments from the config file, config is a yaml file
if args.config is not None:
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise ValueError("No config file specified")

BASE_MODEL_PATH = config['base_model']
DATASET_PATH = args.dataset if args.dataset is not None else config['dataset']
MAX_LENGTH = args.max_length if args.max_length is not None else config['max_length']

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
tokenizer._tokenizer.post_processor = template_gen(tokenizer)

# Load LoRA model and merge with the base model
print("====== Loading the LoRA adapter ======")
model = PeftModel.from_pretrained(model, args.adapter_path)
model.to(device)
model.eval()


# Load the dataset
print("====== Loading dataset ======")
dataset = load_dataset('parquet', data_files=DATASET_PATH, split='train')
gen_output = []
for gen_input in tqdm(dataset['text_gen']):
    inputs = tokenizer(gen_input, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=MAX_LENGTH, do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.05,
        num_return_sequences=1,
    )
    gen_output.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

df = pd.read_parquet(DATASET_PATH)
df['gen_output'] = gen_output
df.to_parquet(os.path.join(os.path.dirname(DATASET_PATH), 'gen_output.parquet'))