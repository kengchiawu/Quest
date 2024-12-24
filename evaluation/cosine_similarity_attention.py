import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention

def compute_head_cosine_similarity(attn_weights, output_file, layer_id):
    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()
    
    # 选择特定位置的注意力权重，选择最后一列query
    attn_slice = attn_weights[:, :, -1, :]  # Shape: (bsz, num_heads, kv_seq_len)
    
    # 打开文件准备追加写入
    with open(f"{args.output_dir}/cos_sim.txt", 'a') as f:
        for i in range(bsz):
            head_i = attn_slice[i]  # Shape: (num_heads, kv_seq_len)
            
            # 计算head之间的余弦相似度
            cos_sim_matrix = torch.zeros((num_heads, num_heads))
            for j in range(num_heads):
                for k in range(j, num_heads):
                    sim = F.cosine_similarity(head_i[j].unsqueeze(0), head_i[k].unsqueeze(0), dim=1)
                    cos_sim_matrix[j, k] = sim
                    if j != k:
                        cos_sim_matrix[k, j] = sim  # 对称填充
            
            # 将相似度矩阵转换为字符串并写入文件
            cos_sim_str = '\n'.join([','.join([f"{val:.6f}" for val in row]) for row in cos_sim_matrix.tolist()])
            f.write(f"q_len:{q_len}\nLayer_id {layer_id}:\n{cos_sim_str}\n\n")


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # New cache format
    if isinstance(past_key_value, DynamicCache):
        kv_seq_len = past_key_value.get_seq_length()
    # Legacy cache format
    else:
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            assert isinstance(past_key_value, tuple)
            kv_seq_len += past_key_value[0].shape[-2]
    
    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    # New cache format
    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    # Legacy cache format
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        

    attn_weights_for_selection = attn_weights


    mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    #attn_weights.size() = (bsz, self.num_heads, q_len, kv_seq_len)
    (bsz_sim, _ , q_len_sim, kv_seq_len_sim) = attn_weights.size()
    if q_len_sim % 500 == 499:
        compute_head_cosine_similarity(attn_weights, self.output_dir, self.layer_id)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    #attn_output.size() = (bsz, q_len, self.num_heads, self.head_dim)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


#global layer_id
#layer_id = 32

def enable_head_cos_sim_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        #global layer_id
        if isinstance(module, (LlamaAttention, MistralAttention)):
            # For longchat model
            #idx_offset += 1
            model._modules[name].layer_id = model._modules[name].layer_idx
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].output_dir = args.output_dir
