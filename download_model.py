import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
from huggingface_hub import login
import torch
login()
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
model_name_or_path = "/dataset/model/longchat/longchat-7b-v1.5-32k"#"meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    #cache_dir="~/root/autodl-tmp",
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)