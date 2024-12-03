import argparse
from argparse import ArgumentParser
import random
import re
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
import torch
import warnings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)
from transformers.models.mistral.modeling_mistral import MistralAttention

def enable_quest_attention_layer_count(model,args):
    for name,module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_layer_count(module,args)
        if isinstance(module, (LlamaAttention, MistralAttention)):
           print( model._modules[name].layer_idx)

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model = args.model
    torch.cuda.empty_cache()
    if 'llama' in model.lower():
        enable_tuple_kv_cache_for_llama()
    loaded = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Enabling Quest Attention")
    enable_quest_attention_layer_count(loaded,args)

def add_args(parser: ArgumentParser):
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--dynamic-part-ntk", action="store_true")
    parser.add_argument("--dynamic-yarn", action="store_true")
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--part-ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--yarn", type=float)
    parser.add_argument("--rerope", type=float)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--gpt-neox-max-length", type=int)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--original-max-position-embeddings", type=int)
    parser.add_argument("--sliding-window-attention", type=int)
    parser.add_argument("--custom-model", action="store_true")
    parser.add_argument("--custom-model-together", action="store_true")
    parser.add_argument("--custom-model-mistral", action="store_true")
    parser.add_argument("--no-use-cache", action="store_true")
    return parser


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",type=str, default="lmsys/longchat-7b-v1.5-32k")
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--length-step", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-file", type=str)

    parser.add_argument("--quest", action="store_true", help="Enable quest attention")
    parser.add_argument("--token_budget", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=16)
    main(add_args(parser).parse_args())
