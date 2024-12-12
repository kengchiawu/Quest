import torch
from tqdm import tqdm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import argparse
from argparse import ArgumentParser

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", action="append", nargs="+")
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--fixed-length", type=int)
parser.add_argument("--max-tokens", type=int, default=8192)
parser.add_argument("--min-tokens", type=int, default=256)
parser.add_argument("--tokens-step", type=int)
parser.add_argument("--length-step", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output_dir", type=str)

parser.add_argument("--num_eval_tokens", type=int, default=None)
#h2o parameters
parser.add_argument('--h2o', action='store_true',help="Enable heavy-hitter attention")
parser.add_argument("--heavy_ratio", type=float, default=0.1)
parser.add_argument("--recent_ratio", type=float, default=0.1)
#quest parameters
parser.add_argument("--quest", action="store_true", help="Enable quest attention")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")

    #rope_scaling_config = {
    #    "type": "llama3",  # 或者其他支持的类型
    #    "factor": 32.0     # 这里使用 factor 作为缩放因子
    #}

    # however, tensor parallel for running falcon will occur bugs
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        #cache_dir=args.cache_dir,
        #rope_scaling=rope_scaling_config,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer,config


args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")

model, tokenizer,config = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.quest:
    print("Enable quest attention")
    from evaluation.quest_attention import (
        enable_quest_attention_eval,
    )

    enable_quest_attention_eval(model, args)

if args.h2o:
    print('Enable heavy hitter')
    from evaluation.h2o_modify_llama import(
        convert_kvcache_llama_heavy_recent,
    )
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio
    #checkpoint = copy.deepcopy(model.state_dict())
    model = convert_kvcache_llama_heavy_recent(model, config)
    #model.load_state_dict(checkpoint)
    model.half().eval().cuda()

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][:1]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
